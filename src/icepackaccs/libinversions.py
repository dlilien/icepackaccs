#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2026 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

import contextlib
import ctypes
import os
import sys
import threading

import firedrake


def _flush_all_output():
    """Flush Python and C stdio before redirecting process output.

    Written by Codex, needed as helper function for capture_rank0_output."""
    sys.stdout.flush()
    sys.stderr.flush()
    ctypes.CDLL(None).fflush(None)


@contextlib.contextmanager
def capture_rank0_output(log_path=None, line_callback=None):
    """Log and print only rank 0 output while suppressing other ranks.

    Usage:
    with capture_rank0_output(log_fn, record_rol_progress):
        do something (usually estimator.solve())
    The log_fn will store stdout (only written off by rank 0) and stderr (written off by all ranks).
    The line_callback will be called on each line of output from rank 0. For example, cature the loss function.
    Both are optional.

    Written by Codex since this is non-scientific code and confusing to write,
    tested by David Lilien on 5/29/2026."""
    comm = firedrake.COMM_WORLD
    rank = comm.rank
    comm.Barrier()

    if rank != 0:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            _flush_all_output()
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
            comm.Barrier()
        return

    read_fd, write_fd = os.pipe()
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    def reader():
        with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
            if log_path is None:
                for line in pipe:
                    os.write(old_stdout, line.encode("utf-8"))
                    if line_callback is not None:
                        line_callback(line)
            else:
                with open(log_path, "w", encoding="utf-8") as log:
                    for line in pipe:
                        log.write(line)
                        log.flush()
                        os.write(old_stdout, line.encode("utf-8"))
                        if line_callback is not None:
                            line_callback(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    _flush_all_output()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    try:
        yield
    finally:
        _flush_all_output()
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        thread.join()
        os.close(old_stdout)
        os.close(old_stderr)
        comm.Barrier()
