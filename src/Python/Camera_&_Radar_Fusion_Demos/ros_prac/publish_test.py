import time
import roslibpy

client = roslibpy.Ros(host='localhost', port=9090)
client.run()

person_pub = roslibpy.Topic(client, '/person_detected', 'std_msgs/Bool')
dist_pub   = roslibpy.Topic(client, '/person_distance', 'std_msgs/Float32')

try:
    while True:
        # demo: toggle values
        person_pub.publish(roslibpy.Message({'data': True}))
        dist_pub.publish(roslibpy.Message({'data': 1.5}))
        time.sleep(3)

        person_pub.publish(roslibpy.Message({'data': True}))
        dist_pub.publish(roslibpy.Message({'data': 3.0}))
        time.sleep(3)

        person_pub.publish(roslibpy.Message({'data': False}))
        dist_pub.publish(roslibpy.Message({'data': 0.5}))
        time.sleep(3)

finally:
    person_pub.unadvertise()
    dist_pub.unadvertise()
    client.terminate()