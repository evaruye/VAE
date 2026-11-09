# Gaussian calculation inputs

This directory contains 34 mapped calculation pairs (68 `.gjf` files). The table below links these pairs to the 39 DFT-bearing entries in `Raw_Material_Descriptors.xlsx`, sheet `Descriptors`. Excel row numbers include the header row.

## File use

- `<prefix>.gjf` contains the geometry-optimization input.
- `<prefix>_high.gjf` contains the corresponding higher-level single-point/property input with explicit coordinates.
- Use the method, basis set, solvent model, charge, multiplicity, and options recorded in each individual file. Optimization settings vary among species.
- Each input includes Cartesian coordinates and does not require a pre-existing checkpoint through `Geom=Check`, `Guess=Read`, or `%oldchk`. `%chk` names the checkpoint written by that job.
- These are calculation inputs. Gaussian output logs and descriptor-extraction scripts are not included in this directory. Descriptor definitions are in `../Data_Dictionary.xlsx`; calculation settings are specified in the individual Gaussian input files.

## Material-to-file mapping

The formula and charge below refer to the species in the input file, which may be an ion rather than the complete commercial salt. The paired single-point filename is obtained by adding `_high` before `.gjf`.

| Excel row | Material identifier | Optimization input | Formula | Charge | Multiplicity |
| --- | --- | --- | --- | ---: | ---: |
| 2 | tert-Butyl hydroperoxide | [TBHP-1.gjf](TBHP-1.gjf) | C4H10O2 | 0 | 1 |
| 3 | Ammonium persulfate | [Pps-1.gjf](Pps-1.gjf) | O8S2 | -2 | 1 |
| 4 | Potassium persulfate | [Pps-1.gjf](Pps-1.gjf) | O8S2 | -2 | 1 |
| 5 | Hydrogen peroxide | [H2O2-2.gjf](H2O2-2.gjf) | H2O2 | 0 | 1 |
| 6 | Sodium persulfate | [Pps-1.gjf](Pps-1.gjf) | O8S2 | -2 | 1 |
| 7 | Sodium metabisulfite | [Sd.gjf](Sd.gjf) | O5S2 | -2 | 1 |
| 8 | Sodium erythorbate | [isona-13.gjf](isona-13.gjf) | C6H7O6 | -1 | 1 |
| 9 | Zinc formaldehyde sulfoxylate | [Sfs-12.gjf](Sfs-12.gjf) | CH3O3S | -1 | 1 |
| 10 | Sodium formaldehyde sulfoxylate | [Sfs-12.gjf](Sfs-12.gjf) | CH3O3S | -1 | 1 |
| 11 | Sodium acetone bisulfite | [ASb-1.gjf](ASb-1.gjf) | C3H7O4S | -1 | 1 |
| 12 | L-Ascorbic acid | [VC-4.gjf](VC-4.gjf) | C6H8O6 | 0 | 1 |
| 13 | 1,1,3,3-Tetramethyldisiloxane | [Td.gjf](Td.gjf) | C4H14OSi2 | 0 | 1 |
| 14 | Tartaric acid | [ta_8.gjf](ta_8.gjf) | C4H6O6 | 0 | 1 |
| 15 | Sodium thiosulfate | [St-1.gjf](St-1.gjf) | O3S2 | -2 | 1 |
| 16 | Sodium bisulfite | [Sbe-1.gjf](Sbe-1.gjf) | HO3S | -1 | 1 |
| 17 | Glyoxal sodium bisulfite | [Sgb-11.gjf](Sgb-11.gjf) | C2H4O8S2 | -2 | 1 |
| 18 | FF6 | [FF6-13.gjf](FF6-13.gjf) | C2H2O5S | -2 | 1 |
| 19 | E18 | [FF6-13.gjf](FF6-13.gjf) | C2H2O5S | -2 | 1 |
| 20 | FF7 | [FF6-13.gjf](FF6-13.gjf) | C2H2O5S | -2 | 1 |
| 21 | Vinyltrimethoxysilane | [Vts-2.gjf](Vts-2.gjf) | C5H12O3Si | 0 | 1 |
| 22 | 1,6-Hexanedioldiacrylate | [16-H6.gjf](16-H6.gjf) | C12H18O4 | 0 | 1 |
| 23 | N-(Hydroxymethyl)acrylamide | [NMA-11.gjf](NMA-11.gjf) | C4H7NO2 | 0 | 1 |
| 24 | N-Succinimidyl acrylate | [NSa-6.gjf](NSa-6.gjf) | C7H7NO4 | 0 | 1 |
| 25 | Acrylamide | [2-Pp1.gjf](2-Pp1.gjf) | C3H5NO | 0 | 1 |
| 26 | Crotonic acid | [3-Ma2.gjf](3-Ma2.gjf) | C4H6O2 | 0 | 1 |
| 27 | Acrylic acid | [Pa-2.gjf](Pa-2.gjf) | C3H4O2 | 0 | 1 |
| 28 | N-(4,4-diethoxybutyl)prop-2-enamide | [N4P2E-5.gjf](N4P2E-5.gjf) | C11H21NO3 | 0 | 1 |
| 29 | VeoVa10 | [VeoVa10-1.gjf](VeoVa10-1.gjf) | C12H22O2 | 0 | 1 |
| 30 | VeoVa11 | [VeoVa11-1.gjf](VeoVa11-1.gjf) | C13H24O2 | 0 | 1 |
| 31 | Sodium vinyl sulfonate | [Sv-1.gjf](Sv-1.gjf) | C2H3O3S | -1 | 1 |
| 32 | Butyl Acrylate | [nBa-10.gjf](nBa-10.gjf) | C7H12O2 | 0 | 1 |
| 33 | N-(Isobutoxymethyl)acrylamide | [N-IA2.gjf](N-IA2.gjf) | C8H15NO2 | 0 | 1 |
| 34 | Vinyltriethoxysilane | [Vtts-10.gjf](Vtts-10.gjf) | C8H18O3Si | 0 | 1 |
| 35 | Diacetone acrylamide | [DAAm-1.gjf](DAAm-1.gjf) | C9H15NO2 | 0 | 1 |
| 36 | N-(Butoxymethyl)acrylamide | [N-Bm2.gjf](N-Bm2.gjf) | C8H15NO2 | 0 | 1 |
| 37 | Glycidyl methacrylate | [SYM-5.gjf](SYM-5.gjf) | C7H10O3 | 0 | 1 |
| 38 | Hydroxyethyl acrylate | [Ha-1.gjf](Ha-1.gjf) | C5H8O3 | 0 | 1 |
| 39 | EHA | [2-EA1.gjf](2-EA1.gjf) | C11H20O2 | 0 | 1 |
| 40 | Itaconic acid | [2-Ma2.gjf](2-Ma2.gjf) | C5H6O4 | 0 | 1 |

## Shared molecular representations

- Ammonium, potassium, and sodium persulfate map to `Pps-1`, representing the persulfate dianion.
- Zinc and sodium formaldehyde sulfoxylate map to `Sfs-12`, representing the same hydroxymethanesulfinate anion. Their isotropic polarizability entries are both 9.8261281201 in the descriptor table.
- FF6, E18, and FF7 map to the supplied `FF6-13` representative structure. This correspondence does not imply identical commercial compositions.
- `NSa-6` is N-Succinimidyl acrylate (C7H7NO4). `N4P2E-5` is N-(4,4-diethoxybutyl)prop-2-enamide (C11H21NO3). The VeoVa11 entry uses the supplied C13H24O2 representative structure in `VeoVa11-1`.
