"""
RPN (Reverse Polish Notation) PDE compiler and learned token embeddings.

Overview
~~~~~~~~

This module provides:

1. **Compiler** (:mod:`qg.solver.opt.operator.rpn.compiler`)
   - Parses RPN token sequences into executable PDE components
   - Maintains linear/nonlinear separation for spectral solver

2. **Embeddings** (:mod:`qg.solver.opt.operator.rpn.embeddings`)
   - Fixed vocabulary mapping token→ID→learned vector
   - Category-aware embedding (variable, operator, constant, …)
   - Scalar‑valued constants via small MLP (no discretisation)

3. **Algebra rules** (:mod:`qg.solver.opt.operator.rpn.algebra`)
   - Sound mathematical rewrites on token‑ID tensors
   - Used for contrastive‑learning positives (InfoNCE)
   - Modular: arithmetic, Jacobian, calculus, trigonometric, etc.

4. **Contrastive training** (:mod:`qg.solver.opt.operator.rpn.contrastive`)
   - Single‑encoder (RPN string → pooled embedding → projection)
   - Positive pairs from algebraic equivalence
   - Negative pairs from other batch sequences (InfoNCE)

5. **Generator** (:mod:`qg.solver.opt.operator.rpn.generator`)
   - Random RPN expression generator for training data
   - Configurable complexity and operator distribution
   - Supports scalar parameters and constants

All modules are designed for ``torch`` tensors and GPU acceleration.
"""

from .compiler import RPNCompiler

def compile_pde_rpn(rpn, derivative, pde_params):
    """Legacy alias for backwards compatibility."""
    return RPNCompiler(derivative, pde_params).compile(rpn)

# Re-export for cleaner API
__all__ = [
    # Compiler
    "RPNCompiler",
    "compile_pde_rpn",
]
