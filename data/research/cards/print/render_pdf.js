const { chromium } = require('playwright');
(async () => {
  const src = process.argv[2] || 'wk12_2026_thu_card_v4_broadcast.html';
  const out = process.argv[3] || '../wk12_2026_thu_card_v4_broadcast.pdf';
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 816, height: 1056 }, deviceScaleFactor: 2 });
  await page.goto('file://' + process.cwd() + '/' + src);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  const report = await page.evaluate(() => Array.from(document.querySelectorAll('.page')).map((p, i) => {
    const pageBottom = p.getBoundingClientRect().bottom;
    const kids = Array.from(p.children).filter(c => !c.classList.contains('foot'));
    const contentBottom = Math.max(...kids.map(c => c.getBoundingClientRect().bottom));
    const clipped = Array.from(p.querySelectorAll('.lt .body')).filter(b => b.scrollWidth > b.clientWidth + 1).length;
    return { page: i + 1, spare_px: Math.round(pageBottom - contentBottom - 30), clipped_bars: clipped };
  }));
  console.log(JSON.stringify(report));
  const pages = await page.locator('.page').all();
  for (let i = 0; i < pages.length; i++) {
    await pages[i].screenshot({ path: '/tmp/prism_card_p' + (i + 1) + '.png' });
  }
  await page.pdf({ path: out, width: '8.5in', height: '11in', printBackground: true, margin: { top: 0, right: 0, bottom: 0, left: 0 } });
  await browser.close();
})();
