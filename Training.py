from Models.MobileNet import MobileNet, rpn, Classifier, pose, roi_pooling
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input
from Helpers.config import Config
from Helpers import Losses


config = Config()
classes_count = 21

img_input = Input(shape=(None, None, 3))
roi_input = Input(shape=(config.num_rois, 4))


num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
shared = MobileNet(img_input, roi_input, num_anchors=num_anchors, classes_count=classes_count)


# rpn model defining
rpn_model = rpn(img_input, shared[0], num_anchors)
optimizer_rpn = keras.optimizers.Adam(lr=1e-4)
rpn_model.compile(optimizer=optimizer_rpn, loss={"rpn_out_class": Losses.rpn_loss_cls(num_anchors), "rpn_out_regress": Losses.rpn_loss_regr(num_anchors)})





# classifier model defining
classifier_model = Classifier(img_input, shared[0], roi_input, config.num_rois, nb_classes=21, trainable=True)
optimizer_classifier = keras.optimizers.Adam(lr=1e-4)
classifier_model.compile(optimizer=optimizer_classifier, loss={"dense_class_21": Losses.class_loss_cls, "dense_regress_21": Losses.class_loss_regr(classes_count -1)})

#pose model defining
pose_model = pose(img_input, roi_input, config.num_rois, pool_size=7, num_classes=classes_count)
optimizer_pose = keras.optimizers.Adam(lr=1e-4)



rpn_value = rpn_model(img_input)
classifier_value = classifier_model([img_input, roi_input])
pose_value = pose_model([img_input, roi_input])



# complete model defining
model_all = keras.Model([img_input, roi_input], [rpn_value + classifier_value, pose_value])
model_all.compile(optimizer="adam", loss="mae")

model_all.summary()






