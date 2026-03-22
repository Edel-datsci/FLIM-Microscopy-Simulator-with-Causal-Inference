"""
ED-MRDT v1.1 — Event-Driven Membrane Receptor Dynamics & Translucent imaging
=============================================================================
Simulator: membrane receptor-ligand dynamics → FLIM/FRET synthetic images.
Pipeline: Brownian Dynamics → CTMC Photophysics → Virtual Microscope → FLIM
"""
__version__ = "1.1.0"

from .config import SimulationConfig, load_config
from .particles import ParticleState
from .dynamics import DynamicsEngine
from .photophysics import PhotonSimulator, PhotonBatch
from .microscope import VirtualMicroscope, FLIMFrame
from .pipeline import SimulationPipeline, SimulationResult, FrameResult, GroundTruth
from .io import HDF5Writer, write_results, read_metadata, read_flim_stack
