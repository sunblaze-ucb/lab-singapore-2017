import collections
import contextlib
import time


class Report(object):
    """A simple way to generate reports."""

    def __init__(self, filename):
        self.filename = filename
        self._report = collections.OrderedDict()
        self._start = time.time()

    def add(self, name, value):
        self._report[name] = value

    def extend(self, values):
        self._report.update(values)

    @contextlib.contextmanager
    def add_time_block(self, name):
        """Record how long a block of code takes."""
        start = time.time()
        yield
        duration = time.time() - start
        self._report[name] = duration

    def save(self):
        # Record overall duration.
        self._report['time_overall'] = time.time() - self._start

        with open(self.filename, 'w') as output:
            for name, value in self._report.items():
                output.write('{}\t{}\n'.format(name, value))
