"""Convenience import for the public decision contract.

The implementation lives in exp_protocol, which both scientist and private
checkouts already ship. No new file is added to historical frozen ship lists.
"""

from .exp_protocol.decisions import *
