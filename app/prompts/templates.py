"""Prompt template dataclass and validation."""

from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    """A reusable prompt template with validation.

    Attributes:
        id: Unique identifier (e.g., "scoring.article")
        template: Prompt text with {variable} placeholders
        variables: List of required variable names
        description: Human-readable description
        version: Version string for future A/B testing
    """

    id: str
    template: str
    variables: list[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0.0"

    def format(self, **kwargs) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Formatted prompt string

        Raises:
            KeyError: If required variable is missing
        """
        # Validate all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise KeyError(
                f"Missing required variables for prompt '{self.id}': {missing}"
            )

        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f"PromptTemplate(id='{self.id}', version='{self.version}')"