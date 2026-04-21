from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_main_script_prints_expected_sections(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main.py"),
                "--seed",
                "42",
                "--items",
                "8",
                "--generations",
                "20",
                "--stagnation",
                "5",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for label in (
            "Цены:",
            "Целевая сумма:",
            "Скрытый вектор:",
            "Режим solver:",
            "NGA режим:",
            "Точки staged NGA:",
            "Сила staged NGA, %:",
            "Лучшее решение:",
            "Fitness:",
            "NGA использован:",
            "Поколения NGA:",
            "Количество рестартов:",
            "Rescue использован:",
            "Причина остановки:",
        ):
            self.assertIn(label, completed.stdout)


if __name__ == "__main__":
    unittest.main()
