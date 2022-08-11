The python file included contains the entirety of the Hough Detector implementation for circles.

The speed up function uses the gradient direction to reduce the number of pixels used in finding the center of the circle.

Here are the results of the detection:

![alt text](https://media.discordapp.net/attachments/793105657145720862/1007284628698181732/unknown.png)

Worth noting that the speed up function has an effect of about 70% on the computation time taken.
