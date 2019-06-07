# Image Classification and Model Interpretability

The purpose of this project is to explain the decisions of a Convolutional Neural Network.
If we are trying to build a model to make important decisions, like in healthcare, we probably want to know the reasons why the model took its decisions. A good accuracy or recall is not guarantee that the model distinguish the classes in the right way. Thus, we will try to investigate what patterns have been learned by our model. 

This project is about detecting pneumonia or no pneumonia from x-ray images of pediatric patients and try to explain the predictions of the Network. The explanations from the model was compared to Medical Doctors' statements. 

Examples: We trained our model (MobileNet Fine-Tuned) and achieved 99.4% Recall. 
* Our network is 100.00% sure the below picture is **PNEUMONIA** and it is right. Below we can see the explanation of a MD and of the Grad-CAM explainer and Lime explainer which we used.

![An example of Peumonia](https://user-images.githubusercontent.com/31864574/59109088-146dce00-8945-11e9-8528-bac49b832b42.png)
Left: the original X-ray.  Right: the MD's quick diagnose.

![2](https://user-images.githubusercontent.com/31864574/59109434-b55c8900-8945-11e9-8254-6ba6d81c27c8.png)
Left: Grad-CAM explainer.  Right: Lime explainer.

We can see that the Grad-CAM explainer shows that the network believes that the area near R (which indicates the right side of the X-ray) is important which is misleading information according to MD’s diagnose.


* Our network is 99.48% sure the bellow picture is **NORMAL** and it is right. Bellow we can see the explanation of the Grad-CAM explainer and Lime explainer which we used.
We can see that the explainers indicate that the whole lungs are important to take the decision.

![norm](https://user-images.githubusercontent.com/31864574/59113868-94e4fc80-894e-11e9-934f-1cb8d15c665f.png)
Left: the original X-ray.  Right: Grad-CAM explainer.

![norm1](https://user-images.githubusercontent.com/31864574/59114140-30766d00-894f-11e9-8b95-05cd05f21169.png)
Left: Lime explainer, just the region.  Right: Lime explainer, the whole picture (green region Pros of NORMAL label).

Note: We can definatly make a better model, but from the above two explanations we know that the model is learning.


## Frameworks & Technologies
Keras, Lime, OpenCV, Numpy, Pandas, matplotlib, Glob, google.colab, Kaggle

---

## Data
There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. 
[(kaggle link)](https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images)

![d](https://user-images.githubusercontent.com/31864574/59118849-ea72d680-8959-11e9-830e-9372bcdf8f69.png)

## Model
Transfer learning was used, more significantly the MobileNet with the trained weights on imagenet. I did some modifications to the MobileNet to adapt it to this case.  The weights of the first 34 layers were freezed, of total 92 layers.

![MobileNet](https://cdn-images-1.medium.com/max/800/1*XeJGMg7siqgjI6kQ3gke9A.png)

Some details for the model:
*	Reduce Learning Rate On Plateau, monitoring the validation loss 
*	Model Checkpoint, monitoring validation loss and saving the best epoch’s weight 
*	Regularizer l2 and Dropout between Dense layers and BatchNormalization between Convolutional layers to reduce overfitting
*	Adam optimizer (lr=0.001, decay=0.0001)

![learning_curves](https://user-images.githubusercontent.com/31864574/59112418-a2e54e00-894b-11e9-9c5d-c493dc7d049a.png)
learing curves


## Model Explainability
For the network's explanation, two techniques are used:

1. **Gradient-weighted Class Activation Mapping** (Grad-CAM), a technique for producing "visual explanations" for decisions from a large class of CNN-based models, making them more transparent, by Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra.
‘Grad-CAM uses the gradients of any target concept (say logits for ‘dog’ or even a caption), flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.’

Modifying the base network to remove all fully-connected layers at the end, and including a tensor product (followed by softmax), which takes as input the Global-Average-Pooled convolutional feature maps, and outputs the probability for each class.

![GCAM](https://cdn-images-1.medium.com/max/1000/1*8iyCBSx6i2lRpnKLe5bIrg.png)

2. **Local Interpretable Model-agnostic Explanations** (LIME), a method presented by Marco Tulio Ribeiro, Sameer Singh and Carlos Guestrin in 2016.

Lime method works by making alterations on different features on a particular input and seeing which of those alterations make the biggest difference to the output classification. Thus highlighting the features most relevant to the network’s decision. The key to lime’s effectiveness is to local element. That means that it does not try explain all the decisions that a network might make across all possible inputs, only the factors that uses to determine its classification for one particular input.


## More case examples
* Case 1: Our model with 99.4% Recall. 
Here the network is 100.00% sure this x-ray is PNEUMONIA and it is right. This is our first example. Bellow we can see from Lime library the Pros (with Green) and Cons (with red) of the lime explainer, using: positive_only=False and hide_rest=False.
![PNprco](https://user-images.githubusercontent.com/31864574/59115706-839def00-8952-11e9-8e5f-07f642974d98.png)


*  Case 2: Our model with 99.4% Recall
Here the network is 100.00% sure this x-ray is PNEUMONIA and it is right. Below we can see the explanation of a MD and of the Grad-CAM explainer and Lime explainer which we used.

![pn2](https://user-images.githubusercontent.com/31864574/59117252-fc527a80-8955-11e9-89bc-5e3b02247e95.png)
Left: the original X-ray. Right: the MD's quick diagnose.

![pn2_1](https://user-images.githubusercontent.com/31864574/59117883-851de600-8957-11e9-9ee7-a4086eed5ab9.png)
Left: Grad-CAM explainer. Right: Lime explainer.

In this case, both Grad-CAM explainer and Lime explainer indicate wrong areas of interest. 


*  Case 3: Another network, our own CNN architecture with 94.3% Recall, which we thought that was training great but it was creating patterns where was nothing. 
![1Co](https://user-images.githubusercontent.com/31864574/59111367-8ea05180-8949-11e9-966d-1b5027e05462.png)


## Outcomes:
* As we can see the Lime method explains better what a convolutional neural network has seen.
* With more tuning of the hyperparameters and trying different architectures we can achieve better performance.


## More to see
* Use Inception V3 architecture as proposed here: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
* Use the framework Skater for Model Interpretation: [(link)](https://github.com/oracle/Skater)
* Visualize channels in intermediate activations and filters


## References
* Dataset: https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images
* MobileNet: https://arxiv.org/abs/1704.04861
* Grad-CAM: https://arxiv.org/abs/1610.02391
* Francois Chollet - Deep Learning with Python (2017, Manning Publications), Chapter 5
* https://medium.com/@mohamedchetoui/grad-cam-gradient-weighted-class-activation-mapping-ffd72742243a
* http://www.hackevolve.com/where-cnn-is-looking-grad-cam/
* Lime library: https://github.com/marcotcr/lime
* Model-Agnostic Interpretability of Machine Learning https://arxiv.org/abs/1606.05386
* "Why Should I Trust You?": Explaining the Predictions of Any Classifier https://arxiv.org/abs/1602.04938
* https://towardsdatascience.com/understanding-how-lime-explains-predictions-d404e5d1829c


## License
[MIT](https://choosealicense.com/licenses/mit/)

