.PHONY: build_name_overrides names

# 1) Build overrides from your approved final player_form
#    (replace the --final-csv path with the filename you committed)
build_name_overrides:
	python scripts/utils/build_manual_overrides_from_final.py \
		--final-csv data/player_form_fullnames_complete_force.csv \
		--out data/manual_name_overrides.csv

# 2) Apply canonicalization using Ourlads + overrides (overrides win)
names: build_name_overrides
	python scripts/utils/canonical_names.py \
		--player-form data/player_form.csv \
		--roles-ourlads data/roles_ourlads.csv \
		--manual-overrides data/manual_name_overrides.csv \
		--out outputs/player_form_fullnames.csv \
		--force-manual
