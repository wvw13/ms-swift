"""Encoding prompts; no labels or patient metadata are injected."""
IMAGE_TYPES = {'C': 'color fundus photographs', 'O': 'OCT image sequences', 'B': 'combined color fundus and OCT images sequences'}
DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGE_INSTRUCTIONS = {'C': 'Represent this color fundus photograph for ophthalmic visual-feature and case retrieval. Focus on visible surface lesions, vessels, optic disc appearance, and color or pigment changes. Ignore watermarks, patient identifiers, device names, and non-anatomical text overlays. Encode only visible evidence; do not generate text.', 'O': 'Represent these six ordered OCT slices jointly for ophthalmic visual-feature and case retrieval. Focus on retinal layer architecture, reflectivity, fluid, and foveal contour across slices. Ignore watermarks, patient identifiers, device names, and non-anatomical text overlays. Encode only visible evidence; do not generate text.', 'B': 'Represent these ophthalmic images jointly for visual-feature and case retrieval. The first image is a color fundus photograph; the next six are ordered OCT slices. Preserve visible fundus lesions, vessels, optic disc and pigment features, and OCT layer structure, reflectivity, fluid and foveal contour. Ignore watermarks, identifiers, device names, and non-anatomical text overlays. Do not generate text.'}

def instruction(c,mod,text=False):
    if c['instruction']=='I0':return DEFAULT_INSTRUCTION
    if not text:return IMAGE_INSTRUCTIONS[mod]
    if c['text_kind']=='name':return f'Represent this ophthalmic disease name for retrieval by {IMAGE_TYPES[mod]}.'
    return (f'Represent this ophthalmic visual-feature description for retrieval by {IMAGE_TYPES[mod]}. '
            'Preserve lesion morphology, location, distribution, and structural details. Do not generate additional findings.')
def branch(c,mod,images=None,text=None):
    if images is not None:
        if not images:raise ValueError('empty image branch')
        return {'messages':[{'role':'system','content':instruction(c,mod)},
            {'role':'user','content':'<image>'*len(images)}],'images':images}
    if not isinstance(text,str) or not text.strip() or '<image>' in text:raise ValueError('invalid disease text')
    return {'messages':[{'role':'system','content':instruction(c,mod,True)}, {'role':'user','content':text}]}
