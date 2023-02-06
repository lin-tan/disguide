# DisGUIDE: Disagreement-Guided Data-Free Model Extraction

This is the artifact repository for the AAAI 2023 paper: DisGUIDE: Disagreement-Guided Data-Free Model Extraction.

The codebase is based on prior work codebases from DFME. https://github.com/cake-lab/datafree-model-extraction
The DFME authors in turn based their code on the DFAD paper codebase. https://github.com/VainF/Data-Free-Adversarial-Distillation

In order to run the codebase:
1. Add an appropriate teacher model to disguide/checkpoint/teacher/
   1. Publicly available models are available from the original DFAD codebase: https://github.com/VainF/Data-Free-Adversarial-Distillation
2. Name the model cifar10-resnet34_8x.pt with appropriate substitutions for cifar100 and resnet18
3. Run either ./run_cifar-10.sh or ./run_cifar-100.sh as appropriate.
