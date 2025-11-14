---
name: Model Request / Missing Models
about: Report missing AI models or request model integration
title: '[MODEL] '
labels: models, documentation
assignees: ''
---

## Issue Type
- [ ] Original models not available (2020 face swap models)
- [ ] Request for new model integration
- [ ] Model documentation unclear
- [ ] Model download broken

## Model Information
**Model Name:** [e.g., Face Swap, Style Transfer]
**Framework:** [e.g., TensorFlow, PyTorch, ONNX]
**Source:** [e.g., GitHub repo, paper, model zoo]

## Current Situation
Describe the current situation with this model.

**For Original Models:**
The original 2020 TensorFlow face swap models are no longer available because:
- Too large for GitHub (~500MB each)
- Original Google Drive / S3 links inactive
- See [models/README.md](../../../models/README.md) for alternatives

## What You Need
What are you trying to accomplish?

## Alternatives You've Tried
- [ ] Used `brain_simple.py` with OpenCV effects
- [ ] Tried downloading from [specify source]
- [ ] Attempted to use alternative model [specify]
- [ ] Tried training own model

## Proposed Solution
If requesting new model integration:
- Model URL: [link to model]
- License: [e.g., MIT, Apache 2.0]
- Performance: [e.g., 30 FPS on CPU, 80 FPS on GPU]
- Size: [e.g., 50MB]

**Would help with:**
- [ ] Integration code
- [ ] Testing
- [ ] Documentation
- [ ] Performance benchmarks

## Resources
Links to:
- Model repository
- Pre-trained weights
- Training code
- Documentation
- Papers

## Additional Context
Any other information that might be helpful.

---

**Note:** For immediate use, see [models/README.md](../../../models/README.md) for:
- OpenCV effects (no models needed)
- Popular open-source models
- How to add your own models
