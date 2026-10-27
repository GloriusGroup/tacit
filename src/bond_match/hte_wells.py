"""Convert the published HTE files into the ``wells.csv`` read by ``score_hte.py``.

    python src/bond_match/hte_wells.py chemlex Chemlex_Acidamine_Wetlab_Data.xlsx chemlex.csv
    python src/bond_match/hte_wells.py aichemeco all_HTE_with_condition.csv aichemeco.csv

Each well becomes one reaction SMILES with every species of the well (substrates, coupling reagent, additive, base and
solvent) on the reactant side and the expected amide as the product. Chemlex gives the conversion in % (column
``conversion_pct``), AIChemEco the yield as a fraction, which is converted to % (column ``yield_pct``).
"""

import argparse

import pandas as pd


def chemlex(path: str) -> pd.DataFrame:
    d = pd.read_excel(path)
    rxn = d.Acid + "." + d.Amine + "." + d.Reagents + "." + d.Solvent + ">>" + d.Products
    return pd.DataFrame({"reaction_smiles": rxn, "conversion_pct": d.Conversion})


def aichemeco(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return pd.DataFrame({"reaction_smiles": d.text, "yield_pct": 100 * d["yield"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=["chemlex", "aichemeco"])
    parser.add_argument("source", help="published HTE file")
    parser.add_argument("out", help="wells CSV to write")
    args = parser.parse_args()
    {"chemlex": chemlex, "aichemeco": aichemeco}[args.dataset](args.source).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
