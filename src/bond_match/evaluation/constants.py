"""Shared module-level objects for the evaluation package.

Contents: the package logger, the RDKit periodic table handle ``PT``, a
patched ``Chem.MolToSmiles`` (``custom_mol_to_smiles``) that enables dative
bonds by default, RDKit standardization singletons (``_uncharger``,
``_normalizer``, ``_tautomer_enumerator``), and the SMILES normalization
cache ``_normalize_cache``.
"""

import logging
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


"""Evaluator for BondGAT ranking models.

A single Evaluator class with unified iteration (greedy = beam_width=1).
When the model predicts "stop", any previous match is preserved.
"""
logger = logging.getLogger(__name__)
PT = Chem.GetPeriodicTable()
_original_mol_to_smiles = Chem.MolToSmiles
def custom_mol_to_smiles(mol, **kwargs):
    """Wrap ``Chem.MolToSmiles`` to route keyword arguments through ``SmilesWriteParams``.

    Enables ``includeDativeBonds`` by default and maps keyword arguments such
    as ``isomericSmiles`` and ``kekuleSmiles`` to their parameter-object
    equivalents. Installed as a replacement for ``Chem.MolToSmiles`` at import
    time.

    Args:
        mol: RDKit molecule to serialize.
        **kwargs: SMILES writer options; a pre-built ``params`` object is
            passed through unchanged.

    Returns:
        The SMILES string produced by the original ``Chem.MolToSmiles``.
    """
    if "params" in kwargs:
        return _original_mol_to_smiles(mol, **kwargs)

    params = Chem.SmilesWriteParams()
    params.includeDativeBonds = kwargs.get("includeDativeBonds", True)
    arg_map = {
        "isomericSmiles": "doIsomericSmiles",
        "kekuleSmiles": "doKekule",
    }

    keys_to_remove = []
    for key, value in kwargs.items():
        attr_name = arg_map.get(key, key)
        if hasattr(params, attr_name):
            setattr(params, attr_name, value)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        kwargs.pop(key)
    kwargs["params"] = params

    return _original_mol_to_smiles(mol, **kwargs)
Chem.MolToSmiles = custom_mol_to_smiles
_uncharger = rdMolStandardize.Uncharger()
_normalizer = rdMolStandardize.Normalizer()
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_normalize_cache: dict[str, str] = {}
if __name__ == "__main__":
    main()
