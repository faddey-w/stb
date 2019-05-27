import sys
import os
import subprocess
import base64
import logging
import time
import enum
import threading
import queue
from strateobots.ai import base
from collections import defaultdict


log = logging.getLogger(__name__)


class AIModule(base.AIModule):
    def __init__(self, program_storage):
        self.program_storage = program_storage

    def list_ai_function_descriptions(self):
        return [(name, name) for name in self.program_storage.list_program_names()]

    def list_bot_initializers(self):
        return []

    def construct_ai_function(self, team, name):
        program_text = self.program_storage.load_program(name)
        return ProgramBasedFunction(program_text)


class ProgramStorage:
    def __init__(self, directory):
        self.directory = directory

    def list_program_names(self):
        return os.listdir(self.directory)

    def load_program(self, name):
        with open(os.path.join(self.directory, name)) as f:
            return f.read()

    def save_program(self, name, program_text):
        with open(os.path.join(self.directory, name), "w") as f:
            f.write(program_text)


class ProgramBasedFunction:
    def __init__(self, program_code):
        self._program = _program_coroutine(program_code)
        self._initialized = False

    def __call__(self, state):
        if not self._initialized:
            self._program.send(None)
            self._initialized = True

        ok, ctldict, debug = self._program.send(state)
        if not ok:
            raise Exception("User program failed:\n{}".format(debug))
        ctllist = [{"id": bot_id, **ctl} for bot_id, ctl in ctldict.items()]
        return {"controls": ctllist, "debug": debug}


class _MType(enum.Enum):
    CONTROL = "\0\1"
    END = "\0\0"

    def get_content(self, line):
        return line[len(self.value) :]


class _WorkerWrapper:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "strateobots.ai.user_program.worker"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
        )
        self.queue = queue.Queue()
        self.reader_thread = threading.Thread(target=self._reader_thread_main)
        self.reader_thread.setDaemon(True)
        self.reader_thread.start()

    def send(self, text):
        self.proc.stdin.write(base64.b64encode(text.encode("utf-8")) + b"\n")
        self.proc.stdin.flush()

    def receive(self, timeout):
        try:
            return self.queue.get(timeout=timeout).decode("utf-8")
        except queue.Empty:
            return None

    def _reader_thread_main(self):
        for line in self.proc.stdout:
            if line.endswith(b"\n"):
                line = line[:-1]
            self.queue.put(line)


def _program_coroutine(program_txt):
    worker = _WorkerWrapper()
    worker.send(program_txt)
    worker.receive(0.5)

    result = None
    while True:
        state = yield result
        if state is None:
            worker.proc.terminate()
            return
        worker.send(repr(state))

        output = []
        ctls = defaultdict(dict)
        tick_ended = False
        timeout = 0.15
        deadline = time.time() + timeout

        while not tick_ended and time.time() < deadline:
            line = worker.receive(0.05)
            if log.isEnabledFor(logging.DEBUG):
                now = time.time()
                log.debug(
                    "worker: t={:4f} dt={:4f} line={!r}".format(
                        now - deadline + timeout, now, line
                    )
                )
            if line is None:
                continue
            if not line:
                break

            if line.startswith(_MType.CONTROL.value):
                log.debug("interpreted as CONTROL")
                value = _MType.CONTROL.get_content(line)
                bot_id, _, value = value.partition(";")
                bot_id = int(bot_id)
                key, _, value = value.partition(";")
                value = int(value)
                ctls[bot_id][key] = value
            elif line.startswith(_MType.END.value):
                log.debug("interpreted as END")
                tick = int(_MType.END.get_content(line))
                if tick != state["tick"]:
                    log.debug(
                        "old tick (now=%s, got=%s), clearing data", state["tick"], tick
                    )
                    ctls.clear()
                    output = []
                else:
                    log.debug("program done, exiting loop")
                    tick_ended = True
            else:
                log.debug("interpreted as DEBUG")
                output.append(line)

        log.debug("worker: done in {:4f} sec".format(time.time() - deadline + timeout))
        ok = worker.proc.poll() is None
        if not ok:
            output.extend(worker.proc.stdout)
        result = ok, ctls, "\n".join(output)
