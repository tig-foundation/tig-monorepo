from abc import ABC, abstractmethod

Events = dict

class MasterExtension(ABC):
    # what config it expects
    # its name
    # what events it may emit

    @abstractmethod
    def setup(self, config: dict, ctx: Context) -> Events:
        pass

    @abstractmethod
    def on_update(self, ctx: Context, events: Events) -> Events:
        pass

    @abstractmethod
    def on_rpc_call(self) :
        pass

class SlaveExtension(ABC):
    @abstractmethod
    def setup(self, config: dict, ctx: Context) -> Events:
        pass

    @abstractmethod
    def on_update(self, ctx: Context, events: Events) -> Events:
        pass



load config.json as AttrDict
d.extensions is a list of extension names
d.config is a dict of extension name -> dict

loop through extensions, dynamically importing them and instantiating
loop through extensions, call setup


loop through items, finding and instantiating the extension class