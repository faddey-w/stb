# from .vec2d_fc import QualityFunctionModel
# from .shoot_aug import QualityFunctionModel
from .eventbased import QualityFunctionModel
# from .simple import QualityFunctionModel
# from .simple_logexp import QualityFunctionModel
# from .crafted_formula import QualityFunctionModel
from . import eventbased, shoot_aug, simple, vec2d_fc, vec2d_v2, semisparse_fc
from . import vec2d_v3, classic

Model = QualityFunctionModel
