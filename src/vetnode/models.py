from pydantic import BaseModel, ConfigDict


def to_camel(string: str) -> str:
    return ''.join(word.capitalize() for word in string.split('_'))

def to_lower_camel(string: str) -> str:
    camel_string = to_camel(string)
    return string[0].lower() + camel_string[1:]

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_lower_camel,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        validate_assignment=True,
    )
