"""Result and state dataclasses for iterative evaluation of BondGAT ranking models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import torch


class StopReason(Enum):
    """Why an iterative evaluation stopped at a given step."""

    PREDICTED_IS_OVER = "Model selected is_over stop action"
    FORWARD_FAILED = "Forward pass failed (could not rebuild graph)"
    NO_CANDIDATES = "No valid pair candidates found"
    APPLY_FAILED = "Step application failed (invalid BE matrix)"
    MAX_STEPS_REACHED = "Maximum evaluation steps reached"
    NO_PRODUCT = "No valid product from top-1 pair"
    STEP1_FAILED = "Step 1 application failed"

@dataclass
class BEState:
    """State of a bond-electron (BE) matrix during iterative evaluation.

    Attributes:
        be_matrix: Square BE matrix of the current state.
        atoms_identity: Per-atom identity array aligned with the BE matrix.
        reverse_lp_map: Mapping from lone-pair virtual node index to atom index.
        has_catalyst: Whether the sample includes catalyst atoms.
        n_reactant_heavy: Number of heavy atoms belonging to the reactant.
        catalyst_flags: Optional per-node catalyst flags from the source graph.
    """

    be_matrix: np.ndarray
    atoms_identity: np.ndarray
    reverse_lp_map: dict = field(default_factory=dict)
    has_catalyst: bool = False
    n_reactant_heavy: int = 0
    catalyst_flags: Optional[dict] = None

@dataclass
class SampleMeta:
    """Metadata for a single sample extracted during collation.

    Attributes:
        G: NetworkX graph of the reactant.
        mol: RDKit molecule of the reactant.
    """

    reactant_smiles: str
    target_smiles: str
    G: object
    mol: object
    yield_value: Optional[float] = None
    be_matrix: np.ndarray = None
    atoms_identity: np.ndarray = None
    lp_node_map: dict = field(default_factory=dict)
    reverse_lp_map: dict = field(default_factory=dict)
    has_catalyst: bool = False
    n_reactant_heavy: int = 0
    num_steps: int = 1
    step_index: int = 0
    entry_idx: Optional[int] = None
    root_stop_prob: Optional[float] = None

@dataclass
class StepResult:
    """Result of applying a single bond swap step."""

    product_smi: str
    state: Optional[BEState]
    success: bool

@dataclass
class ForwardResult:
    """Result of a model forward pass from a BE state.

    Attributes:
        out: Model output dictionary of PyTorch tensors.
        bond_indices: Bond atom-index pairs for the forward pass.
        state: BE state the forward pass was run from.
        G: NetworkX graph rebuilt from the BE matrix.
        sample: Collate-ready sample dictionary, if available.
    """

    out: dict
    bond_indices: torch.Tensor
    state: BEState
    G: object
    sample: Optional[dict] = None

@dataclass
class Candidate:
    """A scored action candidate from model output."""

    is_over: bool = False
    b1: Optional[tuple] = None
    b2: Optional[tuple] = None
    swap_type: int = 0
    embeddings: Optional[torch.Tensor] = None
    action_score: float = 0.0
    action_prob: float = 0.0
    virtual_prob: float = 0.0
    max_bond_prob: float = 0.0
    pair_score: float = 0.0
    pair_prob: float = 0.0

    def __post_init__(self):
        """Initialize empty embeddings and back-fill action score/probability from pair fields."""
        if self.embeddings is None:
            self.embeddings = torch.empty(0)
        if self.action_score == 0.0 and self.pair_score != 0.0:
            self.action_score = self.pair_score
        if self.action_prob == 0.0 and self.pair_prob != 0.0:
            self.action_prob = self.pair_prob

@dataclass
class PairResult:
    """Result from action scoring (stop or local bond pair)."""

    is_over: bool = False
    local_b1: int = -1
    local_b2: int = -1
    score: float = 0.0
    action_prob: float = 0.0
    stop_prob: float = 0.0

    @property
    def pair_prob(self) -> float:
        """Alias for ``action_prob``."""
        return self.action_prob

@dataclass
class BeamEntry:
    """A single entry in the beam during iterative evaluation."""

    score: float
    smi: Optional[str]
    state: BEState
    history: list
    n_steps: int
    is_match: bool

@dataclass
class TerminalEntry:
    """A terminal beam entry that stopped iterating."""

    score: float
    smi: Optional[str]
    n_steps: int
    is_match: bool
    stop_reason: StopReason

@dataclass
class IterationResult:
    """Final result of iterative evaluation."""

    product_smi: Optional[str]
    predicted_steps: int
    is_match: bool
    stop_reason: StopReason

@dataclass
class EvalResult:
    """Full evaluation result for a single molecule."""

    predicted_product: Optional[str]
    predicted_steps: int
    is_match: bool
    stop_reason: StopReason

@dataclass
class ActiveMolecule:
    """Tracks a molecule through batched multi-step evaluation."""

    batch_idx: int
    meta: SampleMeta
    target_canon: str
    state: BEState
    current_smi: str
    is_match: bool
    history_cache: list
    n_steps: int
    score: float = 0.0

@dataclass
class DisplayResult:
    """Result from the top-K display function."""

    product_smi: Optional[str]
    state: BEState
    is_correct: bool
    is_over: bool = False

