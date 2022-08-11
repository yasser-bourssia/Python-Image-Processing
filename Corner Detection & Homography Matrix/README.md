The python code included contains multiple parts:
- First we start by calculating the different gradient-filtered pictures (on the X and Y directions), then we use that to calculate the Harris Matrix for Corner detection.
- We then use the Harris-detected pixels on both pictures to pair the corners detected in both pictures using the ZMSSD algorithm.
- to find the Homography Matrix, we use the RANSAC algorithm, that is, every pairing of two pixels votes for the best matrix.

Results:

- Harris Detector:

![alt text](https://media.discordapp.net/attachments/793105657145720862/1007287199588433980/unknown.png)

- Pixels pairing:

![alt text](https://media.discordapp.net/attachments/793105657145720862/1007287512445767850/unknown.png)


- RANSAC & Homography Matrix, Panoramic view:


![alt text](https://media.discordapp.net/attachments/793105657145720862/1007287801454284840/unknown.png)
