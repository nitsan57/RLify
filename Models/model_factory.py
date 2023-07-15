from fvcore.common.registry import Registry
models_db = Registry("Models")


def get_model_class(model_type : str):
    model_type = model_type #model_type.lower()
    return models_db.get(model_type)


