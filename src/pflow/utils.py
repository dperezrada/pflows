def log(file: str, task: str, msg: str | int | float | bool) -> None:
    print(f"[{file}.{task}] {msg}")
