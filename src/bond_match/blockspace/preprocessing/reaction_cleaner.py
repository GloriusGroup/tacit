from rdkit import Chem
from rdkit.Chem import rdChemReactions


class ReactionCleaner:

    _coo_pattern: Chem.Mol

    def __init__(self):
        self._coo_pattern = Chem.MolFromSmarts("[CX3:0](=[OX1:1])[OX2:2]")

    def clean_coo_oxygens(self, rxn_smiles):
        rxn = rdChemReactions.ReactionFromSmiles(rxn_smiles)

        for reactant in rxn.GetReactants():
            Chem.SanitizeMol(reactant)
            matches = reactant.GetSubstructMatches(self._coo_pattern)
            for match in matches:
                c_idx, double_o_idx, single_o_idx = match

                c_mapnum = reactant.GetAtomWithIdx(c_idx).GetAtomMapNum()
                single_o_mapnum = reactant.GetAtomWithIdx(single_o_idx).GetAtomMapNum()
                double_o_mapnum = reactant.GetAtomWithIdx(double_o_idx).GetAtomMapNum()

                for product in rxn.GetProducts():
                    Chem.SanitizeMol(product)
                    for atom in product.GetAtoms():
                        if not atom.GetAtomMapNum() == c_mapnum:
                            continue
                        neighbors = {bond.GetOtherAtom(atom).GetAtomMapNum(): (bond.GetOtherAtom(atom), bond.GetBondTypeAsDouble()) for bond in atom.GetBonds()}
                        if single_o_mapnum not in neighbors.keys():
                            continue
                        if neighbors[single_o_mapnum][1] != 2:
                            continue
                        neighbors[single_o_mapnum][0].SetAtomMapNum(double_o_mapnum)
                        if double_o_mapnum in neighbors:
                            neighbors[double_o_mapnum][0].SetAtomMapNum(single_o_mapnum)
        return rdChemReactions.ReactionToSmiles(rxn)

