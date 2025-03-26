
from typing import  Any, ClassVar, List, Tuple, Type, TypeVar, Union
from click import Path
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

from shrike.evaluations.models import EvalConfiguration



class Configuration(BaseSettings):
    
    _yaml_file: ClassVar[Path] = None

    name:str
    evals:List[EvalConfiguration]

    def __init__(self, path=None, *args, **kwargs):
        if path:
            self.insert_file_into_sources(path)
        super().__init__(*args, **kwargs)


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = cls._yaml_file
        return (
            init_settings,
            YamlConfigSettingsSource(
                settings_cls=settings_cls,
                yaml_file=yaml_file,
                yaml_file_encoding="utf-8",
            ),
            env_settings,
            dotenv_settings,
            file_secret_settings
        )