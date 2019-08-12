import pickle
import segmentation_models_pytorch as smp
from tqdm import tqdm

class Validator:
    def __init__(self, model, optimizer, loader, imgsize):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.imgsize = imgsize

        self.results = {"thresholds": [], "iou": [], "f-score": [],
                        "pred_pixels": [], "label_pixels": []}

    @staticmethod
    def iou_metric(pred,label,thr):
        f = smp.utils.metrics.IoUMetric(eps=1., threshold=thr, activation=None)
        return f(pred,label)

    @staticmethod
    def fscore_metric(pred,label,thr):
        f = smp.utils.metrics.FscoreMetric(eps=1., threshold=thr, activation=None)
        return f(pred,label)

    def run(self, steps, device="cpu"):
        self.model.eval()
        self.model.to(device)
        thresholds = [t/float(steps) for t in range(steps)]
        self.results["thresholds"] = thresholds
        with tqdm(self.loader) as iterator:
            for image, label in iterator:
                image = image.to(device)
                label = label.to(device)
                pred = self.model.predict(image)
                for k in range(len(pred)):
                    self._eval_prediction(pred[k], label[k], steps)

    def _eval_prediction(self, pred, label, steps):

        score = [Validator.fscore_metric(pred, label, t/float(steps)).item()
                 for t in range(steps)]
        iou = [Validator.iou_metric(pred, label, t/float(steps)).item()
               for t in range(steps)]
        ppix = [(pred>t/float(steps)).sum().item()/(self.imgsize*self.imgsize)
                for t in range(steps)]
        lpix = [label.sum().item()/(self.imgsize*self.imgsize)]
        self.results["f-score"].append(score)
        self.results["iou"].append(iou)
        self.results["pred_pixels"].append(ppix)
        self.results["label_pixels"].append(lpix)

    def write_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.results, file)

