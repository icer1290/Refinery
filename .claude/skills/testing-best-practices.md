# Testing Best Practices for AI Engine

## Key Lessons Learned from Refactoring

### 1. Use Correct Python Command

**Problem**: Using `python` or `python3` directly may not work if dependencies aren't installed in the system Python.

**Solution**: Always use `uv run python` or `uv run pytest` for projects managed by uv.

```bash
# Correct
uv run python -c "from app.module import func"
uv run pytest tests/test_file.py -v

# Wrong (may fail)
python -c "from app.module import func"
python3 -c "from app.module import func"
```

### 2. Check Indentation When Editing Classes

**Problem**: When adding methods to existing classes, the Edit tool may place them at the wrong indentation level (outside the class).

**Solution**:
- Read the file first to understand the class structure
- Match the indentation exactly - methods inside a class need one level of indentation
- Verify by reading the edited section after making changes

```python
# WRONG - method outside class
class MyClass:
    def existing_method(self):
        pass

def new_method(self):  # This is NOT inside the class!
    pass

# CORRECT - method inside class
class MyClass:
    def existing_method(self):
        pass

    def new_method(self):  # Properly indented inside class
        pass
```

### 3. Update Tests After Refactoring Signatures

**Problem**: Refactoring node functions to use `Runtime` context breaks existing tests that pass `session` directly.

**Solution**: Create mock Runtime objects with context:

```python
from unittest.mock import MagicMock
from langgraph.runtime import Runtime
from app.module.context import ModuleContext

# Create mock runtime
mock_runtime = MagicMock(spec=Runtime)
mock_runtime.context = MagicMock(spec=ModuleContext)
mock_runtime.context.session = None  # or your mock session

# Call the node
result = await nodes.some_node(state, mock_runtime)
```

### 4. Update State Keys When Refactoring

**Problem**: Renamed state fields (e.g., `_pending_action` → `pending_action`) cause test assertions to fail.

**Solution**: When refactoring state, do a grep search for all usages and update them:

```bash
# Find all usages
grep -r "_pending_action" app/ tests/
```

### 5. Test Import Early

**Problem**: Syntax errors or import issues may only appear at runtime.

**Solution**: Run a quick import test before running full tests:

```bash
uv run python -c "from app.module import function; print('OK')"
```

## Quick Test Commands

```bash
# Test imports only
uv run python -c "from app.deep_search import run_deep_search; print('OK')"

# Run specific test file
uv run pytest tests/test_deep_search_nodes.py -v

# Run tests matching a pattern
uv run pytest tests/ -k "deep" -v

# Run all tests with summary
uv run pytest tests/ -v --no-header -q
```

## Common Test Patterns

### Mocking LLM Service

```python
class DummyLLMService:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def reasoning_analysis(self, system_prompt, user_prompt, **kwargs):
        result = self.responses[self.calls]
        self.calls += 1
        return result
```

### Mocking Tool Execution

```python
async def fake_execute_tool(_session, tool_name, tool_input):
    return f"{tool_name}:{tool_input['query']}"

monkeypatch.setattr(nodes, "execute_tool", fake_execute_tool)
```

### Testing State Updates

```python
# For nodes that use Annotated[list, operator.add], they return new items only
result = await nodes.tools_node(state, mock_runtime)
assert len(result["tool_history"]) == 1  # Only the new item, not cumulative

# The state itself would have all items after graph execution
```