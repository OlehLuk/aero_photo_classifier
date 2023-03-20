from PIL import ImageColor
from aero_classificator.cv import AeroPhotoClassifier

if __name__ == '__main__':
    class_examples = ["img/sample/forest.png", "img/sample/field.png", "img/sample/road.png"
                      ]
    colors = [ImageColor.getrgb('lime'), ImageColor.getrgb('yellow'), ImageColor.getrgb('red')
              ]

    AeroPhotoClassifier.launch("img/sample/sample.png", class_examples, colors, 50)
    AeroPhotoClassifier.launch("img/sample/sample.png", class_examples, colors, 50,
                               delta=20)
