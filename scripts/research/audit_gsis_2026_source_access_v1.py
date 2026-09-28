#!/usr/bin/env python3
"""Credential-safe NFL GSIS 2026 access/inventory audit.

Research-only. Performs one authorized login attempt with GitHub Actions secrets,
then inventories authenticated GSIS structure without storing credentials,
cookies, tokens, query strings, response bodies, or screenshots.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

START_URL = "https://www.nflgsis.com/GameStatsLive/Auth/"
SCHEDULE_URL = "https://www.nflgsis.com/GameStatsLive/Schedule"
GSIS_HOSTS = {"nflgsis.com", "www.nflgsis.com"}
SAFE_AUTH_SUFFIXES = ("nflgsis.com", "nfl.com", "nfl.net", "okta.com")

CHALLENGE_TERMS = (
    "verification code", "authenticator", "multi-factor", "multifactor",
    "two-factor", "2fa", "one-time code", "otp", "security code",
    "approve sign in", "approve sign-in", "passkey", "captcha",
    "verify your identity", "device approval",
)
BAD_AUTH_TERMS = (
    "bad username or password", "incorrect password", "invalid password",
    "invalid credentials", "wrong password", "couldn't sign you in",
    "could not sign you in", "unauthorized",
)
LOGIN_TERMS = ("sign in", "log in", "login")


def _parts(url: str):
    try:
        return urlsplit(str(url))
    except Exception:
        return urlsplit("")


def _safe_path(url: str) -> str:
    u = _parts(url)
    host = (u.hostname or "").lower()
    if host not in GSIS_HOSTS:
        return ""
    return (u.path or "/")[:500]


def _sanitized_url(url: str) -> str:
    u = _parts(url)
    host = (u.hostname or "").lower()
    if not host:
        return ""
    return f"{u.scheme}://{host}{u.path or '/'}"[:700]


def _safe_metadata_route(url: str) -> dict:
    u = _parts(url)
    host = (u.hostname or "").lower()
    if not host or not any(host == s or host.endswith("." + s) for s in SAFE_AUTH_SUFFIXES):
        return {}
    return {"host": host, "path": (u.path or "/")[:500]}


def _frames(page: Page):
    return list(page.frames)


def _first_visible(page: Page, selectors: tuple[str, ...]):
    for frame in _frames(page):
        for selector in selectors:
            try:
                loc = frame.locator(selector)
                for i in range(min(loc.count(), 12)):
                    item = loc.nth(i)
                    if item.is_visible():
                        return item
            except Exception:
                continue
    return None


def _click_text(page: Page, labels: tuple[str, ...]) -> bool:
    for frame in _frames(page):
        for label in labels:
            candidates = [
                frame.get_by_role("button", name=re.compile(rf"^{re.escape(label)}$", re.I)),
                frame.get_by_role("link", name=re.compile(rf"^{re.escape(label)}$", re.I)),
                frame.locator(f'input[type="submit"][value="{label}"]'),
                frame.locator(f'input[type="button"][value="{label}"]'),
            ]
            for loc in candidates:
                try:
                    if loc.count() and loc.first.is_visible():
                        loc.first.click()
                        return True
                except Exception:
                    continue
    return False


def _body_signal(page: Page) -> str:
    try:
        text = page.locator("body").inner_text(timeout=5000)
    except Exception:
        text = ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _looks_challenge(page: Page) -> bool:
    text = _body_signal(page)
    if any(term in text for term in CHALLENGE_TERMS):
        return True
    return _first_visible(
        page,
        (
            'input[autocomplete="one-time-code"]',
            'input[name*="otp" i]',
            'input[id*="otp" i]',
            'iframe[src*="captcha" i]',
            '[class*="captcha" i]',
        ),
    ) is not None


def _looks_bad_auth(page: Page) -> bool:
    return any(term in _body_signal(page) for term in BAD_AUTH_TERMS)


def _looks_like_login(page: Page) -> bool:
    url = str(page.url).lower()
    password = _first_visible(
        page,
        ('input[type="password"]', 'input[name*="password" i]', 'input[id*="password" i]'),
    )
    auth_url = any(x in url for x in ("/auth", "/login", "/sign-in", "id.nfl.com"))
    return bool(auth_url and (password is not None or any(t in _body_signal(page)[:4000] for t in LOGIN_TERMS)))


def _input_metadata(page: Page) -> list[dict]:
    rows = []
    for fi, frame in enumerate(_frames(page)):
        try:
            loc = frame.locator("input")
            count = min(loc.count(), 40)
        except Exception:
            continue
        for i in range(count):
            try:
                item = loc.nth(i)
                rows.append(
                    {
                        "frame": fi,
                        "visible": bool(item.is_visible()),
                        "type": (item.get_attribute("type") or "")[:40],
                        "name": (item.get_attribute("name") or "")[:100],
                        "id": (item.get_attribute("id") or "")[:100],
                        "autocomplete": (item.get_attribute("autocomplete") or "")[:80],
                    }
                )
            except Exception:
                continue
    return rows[:40]


def _wait_auth_surface(context, current: Page) -> Page:
    for _ in range(8):
        current.wait_for_timeout(750)
        pages = [p for p in context.pages if not p.is_closed()]
        if pages:
            current = pages[-1]
        try:
            current.wait_for_load_state("domcontentloaded", timeout=2000)
        except PlaywrightTimeoutError:
            pass
        if _first_visible(
            current,
            (
                'input[type="email"]',
                'input[name*="email" i]',
                'input[id*="email" i]',
                'input[name*="username" i]',
                'input[id*="username" i]',
                'input[type="password"]',
            ),
        ) is not None:
            break
        if _looks_challenge(current):
            break
    return current


def _email_field(page: Page):
    field = _first_visible(
        page,
        (
            'input[type="email"]',
            'input[name*="email" i]',
            'input[id*="email" i]',
            'input[name*="username" i]',
            'input[id*="username" i]',
            'input[name="Username"]',
            'input[id="Username"]',
        ),
    )
    if field is not None:
        return field
    # Fallback only on a recognized NFL authentication host and only when there
    # is exactly one visible text-like input.
    route = _safe_metadata_route(page.url)
    host = route.get("host", "")
    if not host or "nfl" not in host:
        return None
    visible = []
    for frame in _frames(page):
        try:
            loc = frame.locator('input:not([type="hidden"]):not([type="password"]):not([type="submit"]):not([type="button"]):not([type="reset"]):not([type="checkbox"]):not([type="radio"])')
            for i in range(min(loc.count(), 10)):
                item = loc.nth(i)
                if item.is_visible():
                    visible.append(item)
        except Exception:
            continue
    return visible[0] if len(visible) == 1 else None


def _password_field(page: Page):
    return _first_visible(
        page,
        ('input[type="password"]', 'input[name*="password" i]', 'input[id*="password" i]'),
    )


def _fill_login_once(context, page: Page, user: str, password: str) -> tuple[Page, dict]:
    meta = {
        "start_url": _sanitized_url(page.url),
        "auth_surface": "",
        "email_field_found": False,
        "password_field_found": False,
        "login_trigger_clicked": False,
        "email_submitted": False,
        "password_submitted": False,
        "post_trigger_route": {},
        "post_trigger_title": "",
        "post_trigger_frames": [],
        "post_trigger_inputs": [],
    }

    email = _email_field(page)
    pwd = _password_field(page)
    if email is None and pwd is None:
        meta["login_trigger_clicked"] = _click_text(page, ("Login", "Log In", "Sign In", "Sign in"))
        page = _wait_auth_surface(context, page)
        meta["post_trigger_route"] = _safe_metadata_route(page.url)
        try:
            meta["post_trigger_title"] = page.title()[:200]
        except Exception:
            pass
        meta["post_trigger_frames"] = [
            _safe_metadata_route(f.url) for f in _frames(page) if _safe_metadata_route(f.url)
        ][:20]
        meta["post_trigger_inputs"] = _input_metadata(page)
        if _looks_challenge(page):
            meta["auth_surface"] = "interactive_challenge_before_credentials"
            return page, meta
        email = _email_field(page)
        pwd = _password_field(page)

    if email is not None:
        meta["email_field_found"] = True
        meta["auth_surface"] = "email_or_username"
        email.fill(user)

    if pwd is not None:
        meta["password_field_found"] = True
        meta["auth_surface"] = "direct_email_password"
        pwd.fill(password)
        meta["password_submitted"] = _click_text(
            page, ("Login", "Log In", "Sign In", "Sign in", "Continue", "Next")
        )
        if not meta["password_submitted"]:
            try:
                pwd.press("Enter")
                meta["password_submitted"] = True
            except Exception:
                pass
        page = _wait_auth_surface(context, page)
        return page, meta

    if email is None:
        meta["auth_surface"] = "no_supported_credential_form"
        return page, meta

    meta["email_submitted"] = _click_text(
        page, ("Continue", "Next", "Sign In", "Sign in", "Login", "Log In")
    )
    if not meta["email_submitted"]:
        try:
            email.press("Enter")
            meta["email_submitted"] = True
        except Exception:
            pass
    page = _wait_auth_surface(context, page)
    if _looks_challenge(page):
        meta["auth_surface"] = "interactive_challenge_after_email"
        return page, meta

    pwd = _password_field(page)
    if pwd is None:
        meta["auth_surface"] = "password_field_not_found"
        meta["post_email_route"] = _safe_metadata_route(page.url)
        meta["post_email_inputs"] = _input_metadata(page)
        return page, meta

    meta["password_field_found"] = True
    meta["auth_surface"] = "email_then_password"
    pwd.fill(password)
    meta["password_submitted"] = _click_text(
        page, ("Continue", "Next", "Sign In", "Sign in", "Login", "Log In")
    )
    if not meta["password_submitted"]:
        try:
            pwd.press("Enter")
            meta["password_submitted"] = True
        except Exception:
            pass
    page = _wait_auth_surface(context, page)
    return page, meta


def _collect_links(page: Page) -> list[dict]:
    rows, seen = [], set()
    try:
        anchors = page.locator("a")
        count = min(anchors.count(), 500)
    except Exception:
        return rows
    for i in range(count):
        try:
            a = anchors.nth(i)
            href = a.get_attribute("href") or ""
            if href.startswith("/"):
                href = "https://www.nflgsis.com" + href
            path = _safe_path(href)
            if not path:
                continue
            text = re.sub(r"\s+", " ", (a.inner_text() or "")).strip()[:120]
            key = (text, path)
            if key not in seen:
                seen.add(key)
                rows.append({"text": text, "path": path})
            if len(rows) >= 100:
                break
        except Exception:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("NFLGSIS_USER", "").strip()
    password = os.environ.get("NFLGSIS_PASS", "").strip()
    base = {
        "version": "NFL_GSIS_2026_SOURCE_ACCESS_AUDIT_V1_1",
        "credentials_present": {"NFLGSIS_USER": bool(user), "NFLGSIS_PASS": bool(password)},
        "credentials_recorded_in_artifact": False,
        "cookies_or_storage_state_recorded": False,
        "screenshots_recorded": False,
        "week3_outcomes_used": False,
        "full_slate_changed": False,
        "production_integration_authorized": False,
    }
    if not user or not password:
        base.update(disposition="GSIS_2026_SECRET_CONFIGURATION_MISSING", authenticated=False, schedule_access=False)
        (args.out_dir / "result.json").write_text(json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(base["disposition"])
        return 0

    network = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36"
        )
        page = context.new_page()

        def on_response(response):
            try:
                route = _safe_metadata_route(response.url)
                if not route:
                    return
                req = response.request
                ctype = (response.headers.get("content-type") or "").split(";")[0][:100]
                key = (req.method[:12], route["host"], route["path"], int(response.status), ctype)
                if key not in network and len(network) < 250:
                    network[key] = {
                        "method": key[0], "host": key[1], "path": key[2],
                        "status": key[3], "content_type": key[4],
                    }
            except Exception:
                pass

        page.on("response", on_response)
        try:
            start_resp = page.goto(START_URL, wait_until="domcontentloaded", timeout=30000)
            start_status = int(start_resp.status) if start_resp else None
        except Exception:
            start_status = None
        page.wait_for_timeout(700)
        page, auth_meta = _fill_login_once(context, page, user, password)

        if _looks_challenge(page):
            disposition, authenticated, schedule_access, schedule_meta = (
                "GSIS_2026_INTERACTIVE_AUTH_REQUIRED", False, False, {}
            )
        elif _looks_bad_auth(page):
            disposition, authenticated, schedule_access, schedule_meta = (
                "GSIS_2026_AUTH_FAILED", False, False, {}
            )
        else:
            try:
                schedule_resp = page.goto(SCHEDULE_URL, wait_until="domcontentloaded", timeout=30000)
                schedule_status = int(schedule_resp.status) if schedule_resp else None
            except Exception:
                schedule_status = None
            page.wait_for_timeout(1500)
            challenge = _looks_challenge(page)
            bad_auth = _looks_bad_auth(page)
            still_login = _looks_like_login(page)
            final_path = _safe_path(page.url)
            try:
                title = page.title()[:200]
            except Exception:
                title = ""
            try:
                visible_text_chars = len(page.locator("body").inner_text(timeout=5000))
            except Exception:
                visible_text_chars = 0
            links = _collect_links(page)
            schedule_access = bool(
                not challenge and not bad_auth and not still_login and final_path
                and schedule_status is not None and 200 <= schedule_status < 400
                and visible_text_chars > 50
            )
            authenticated = schedule_access
            if challenge:
                disposition = "GSIS_2026_INTERACTIVE_AUTH_REQUIRED"
            elif bad_auth:
                disposition = "GSIS_2026_AUTH_FAILED"
            elif schedule_access:
                disposition = "GSIS_2026_AUTHENTICATED_INVENTORY_READY"
            else:
                disposition = "GSIS_2026_ACCESS_SURFACE_UNRESOLVED"
            schedule_meta = {
                "status": schedule_status, "final_path": final_path, "title": title,
                "visible_text_chars": visible_text_chars,
                "same_host_link_count": len(links), "links": links,
            }

        result = dict(base)
        result.update(
            disposition=disposition,
            authenticated=authenticated,
            schedule_access=schedule_access,
            start_status=start_status,
            auth=auth_meta,
            schedule=schedule_meta,
            final_browser_route=_safe_metadata_route(page.url),
            network_inventory=list(network.values()),
            network_inventory_count=len(network),
        )
        (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(disposition)
        print(json.dumps({
            "authenticated": authenticated,
            "schedule_access": schedule_access,
            "final_browser_route": result["final_browser_route"],
            "network_inventory_count": len(network),
        }, sort_keys=True))
        context.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
