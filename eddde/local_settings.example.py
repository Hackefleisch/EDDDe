"""Per-machine optional settings for EDDDe.

Copy this file to `eddde/local_settings.py` and fill in the values you need.
`local_settings.py` is gitignored so each developer/machine keeps its own.

Every setting here is OPTIONAL. Anything left unset (or this whole file
absent) just means the dependent method silently does not register; the
rest of the pipeline runs normally.
"""

# Absolute path to the BCL executable, used by B18 BCL::Mol2D.
# Download the prebuilt installer from
#   https://github.com/BCLCommons/bcl/releases
# (currently `bcl-4.3.1-Linux-x86_64.sh`), run it once, then point this
# at the resulting `bcl.exe`. Leave as None (or delete the line) to skip
# B18 entirely.
BCL_BIN = None
# BCL_BIN = "/home/you/bcl-4.3.1-Linux-x86_64/bcl.exe"
