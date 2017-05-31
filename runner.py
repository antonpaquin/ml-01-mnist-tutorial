from main import Network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10])

print("Before training: {success} / {max}".format(success=net.evaluate(test_data), max=len(test_data)))
net.gradient_descent(training_data, 30, 10, .002, 0.6, test_data=test_data)