from fastmcp import FastMCP

mcp = FastMCP("Math Server")


@mcp.tool()
def calculate(first_num: float, second_num: float, operation: str) -> dict:
    """Perform add, sub, mul, div, or mod operations."""
    operation = operation.lower().strip()

    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}
        result = first_num / second_num
    elif operation in ("mod", "modulus"):
        if second_num == 0:
            return {"error": "Modulus by zero is not allowed"}
        result = first_num % second_num
    else:
        return {
            "error": f"Unsupported operation: {operation}",
            "supported_operations": ["add", "sub", "mul", "div", "mod"],
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@mcp.tool()
def modulus(first_num: int, second_num: int) -> int:
    """Return the remainder after dividing first_num by second_num."""
    if second_num == 0:
        raise ValueError("Modulus by zero is not allowed")
    return first_num % second_num


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)