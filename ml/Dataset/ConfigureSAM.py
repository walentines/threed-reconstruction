from SegmentAnything.SAMInference import SamInference
from GroundingDino.GroundingDinoInference import GroundingDinoInference

class ConfigureSAM:
    def __init__(self, sam_checkpoint, model_type='vit_h', grounding_dino_model_id='IDEA-Research/grounding-dino-base', device='cuda:1'):
        grounding_dino_inference = GroundingDinoInference(grounding_dino_model_id, device)
        self.sam_inference = SamInference(grounding_dino_inference, sam_checkpoint, model_type, device)

    def get_sam_inference(self):
        return self.sam_inference

