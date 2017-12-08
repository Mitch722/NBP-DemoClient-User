# NBP-DemoClient-User
A Work In Progress client for the number plate game. This is an adapted version of the [battleships bot](https://test.aigaming.com/Help?url=downloads) from AIGaming.com.

## Packages
This project makes use of several common libraries for image processing and 

## ToDo
Things which need to be done and a fey ideas on what could be done
### Easy

- Make use of sending off multiple moves in one request

### Medium
- Make use of some number plate statistics (for example [this](https://www.gov.uk/vehicle-registration/q-registration-numbers))
- Pull images from the links provided in the `gamestate` instead of using the hard-coded library.

### Hard

- Make the RMSSearch faster
- Use some cutting-edge cv2 trickery (which is huge and a pain to import in python)