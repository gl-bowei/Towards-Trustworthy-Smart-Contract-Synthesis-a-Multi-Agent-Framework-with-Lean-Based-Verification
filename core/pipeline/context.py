from core.pipeline.diagnostic_utils import attack_property_to_text

class PipelineContext:
    def __init__(self, workspace, checkpoint):
        self.workspace = workspace
        self.checkpoint = checkpoint
        self.solidity_code = ""
        self.properties = []
        self.semantic_obligations = checkpoint.load_stage("semantic_obligations") or []
        self.semantic_property_texts = checkpoint.load_stage("semantic_property_texts") or []
        self.semantic_guidance_texts = checkpoint.load_stage("semantic_guidance_texts") or []
        self.security_patches = [] # Properties derived from concrete attacks.
        self.attack_properties = checkpoint.load_stage("attack_properties") or [] # Structured attack-to-property records.
        for record in self.attack_properties:
            text = attack_property_to_text(record)
            if text:
                self.security_patches.append(text)
        self.feedback_constraints = [] # Repair constraints supplied to the coder.
        self.iteration = 0
