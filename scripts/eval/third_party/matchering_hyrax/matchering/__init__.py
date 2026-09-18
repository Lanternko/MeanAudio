# Minimal vendored subset of Matchering 2.0.6 (GPLv3, Sergree): only the Hyrax limiter
# and what it imports. The full package pulls statsmodels etc. for its EQ matching.
from .defaults import Config, LimiterConfig
