# UNet Architecture build

- overlap-tile strategy:

  - image size 256 x 256

  - overlap tile -> input(padded) size 440 // output size 256
    
    ![https://github.com/X-AI-eXtension-Artificial-Intelligence/5th-BASE-SESSION/tree/main/TEAM%20A/UNet/%ED%98%95%EB%B9%88/5%EC%A3%BC%EC%B0%A8](./pic/padded_input.png) ![https://github.com/X-AI-eXtension-Artificial-Intelligence/5th-BASE-SESSION/tree/main/TEAM%20A/UNet/%ED%98%95%EB%B9%88/5%EC%A3%BC%EC%B0%A8](./pic/padded_transformed_input.png)



- train loss (epoch20)

![https://github.com/X-AI-eXtension-Artificial-Intelligence/5th-BASE-SESSION/tree/main/TEAM%20A/UNet/%ED%98%95%EB%B9%88/5%EC%A3%BC%EC%B0%A8](./pic/train_loss.png)

- test_inference

  ![https://github.com/X-AI-eXtension-Artificial-Intelligence/5th-BASE-SESSION/tree/main/TEAM%20A/UNet/%ED%98%95%EB%B9%88/5%EC%A3%BC%EC%B0%A8](./pic/inference.png)
