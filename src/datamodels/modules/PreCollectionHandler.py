from src.datamodels.modules.BaseModule import BaseHandler
from src.datamodels.modules.ModuleConfig import PreCollectionsConfig

class PreCollectionHandler(BaseHandler):

    def __init__(self, handler_config: PreCollectionsConfig):
        super().__init__(handler_config)


