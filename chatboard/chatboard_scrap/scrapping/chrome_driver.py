from re import template
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import time 
from config import SCRAPPER_URL, DATA_DIR, DEBUG
from dagster import get_dagster_logger
from kubernetes import client, config
from kubernetes.client import CoreV1Api
from kubernetes.stream import portforward


def create_selenium_body():
    container = client.V1Container(
        name="selenium",
        image="selenium/standalone-chrome",
        ports=[
            client.V1ContainerPort(container_port=4444, host_port=4444, protocol='TCP'),
            client.V1ContainerPort(container_port=7900, host_port=7900, protocol='TCP'),
        ],
        image_pull_policy='IfNotPresent',
        # resources=client.V1ResourceRequirements(
        #     requests={'memory': '2Gi'},
        # )
        volume_mounts=[client.V1VolumeMount(mount_path='/dev/shm', name='dshm')],
    )
    podname = "selenium-" + str(uuid.uuid4().hex)

    pod = client.V1Pod(
        api_version='v1',
        kind='Pod',
        metadata=client.V1ObjectMeta(name=podname),
        spec=client.V1PodSpec(
            restart_policy='Never', 
            containers=[container],
            volumes=[client.V1Volume(name='dshm', empty_dir=client.V1EmptyDirVolumeSource(medium='Memory', size_limit='2Gi'))],
        ),
    )
    return pod, podname

def read_status(podname, api_instance):
    api_response = api_instance.read_namespaced_pod_status(
        name=podname,
        namespace="default")
    return api_response.status

def delete_selenium_pod(podname, api_instance):
    api_response = api_instance.delete_namespaced_pod(
        name=podname,
        namespace='default',
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5)
    )
    print('Pod deleted. status={}'.format(str(api_response.status)))


def create_pod(api_instance: CoreV1Api, pod):
    # api_response = api_instance.create_namespaced_pod(
    #     body=pod,
    #     namespace='default',
    # )
    api_response = api_instance.create_namespaced_pod(
        body=pod,
        namespace='default',
    )
    print('Pod created. status={}'.format(str(api_response.status)))
    return api_response

def get_pod_ip(podname, api_instance):
    for _ in range(100):
        api_response = api_instance.read_namespaced_pod_status(
            name=podname,
            namespace="default")

        if api_response.status.phase == 'Waiting':
            get_dagster_logger().info("Waiting for selenium pod to start")
        if api_response.status.phase == 'Terminated':
            get_dagster_logger().info("Selenium pod terminated")
            raise Exception("selenium pod terminated")
        if api_response.status.pod_ip is not None and api_response.status.phase == "Running":
            return api_response.status.pod_ip
        time.sleep(3)
    raise Exception("Pod IP not found")

class ChromeDriver:

    def __init__(self, logger=None):
        self.logger = logger if logger else get_dagster_logger()
        self.driver = None
        self.container = None
        self.podname = None

    def start_container(self):
        try:
            if DEBUG:
                config.load_kube_config()
            else:
                config.load_incluster_config() 
            self.api_instance = client.CoreV1Api()
            pod, podname = create_selenium_body()
            self.podname = podname
            self.pod = pod
            create_response = create_pod(self.api_instance, pod)
            selenium_ip = get_pod_ip(podname, self.api_instance)
            # selenium_ip='localhost'
            # pf = portforward(
            #     self.api_instance.connect_get_namespaced_pod_portforward,
            #     self.podname, 'default',
            #     ports='4444',
            # )
            time.sleep(5)
            for _ in range(3):
                try:
                    # chrome_options = webdriver.ChromeOptions()
                    # chrome_options.add_argument('--no-sandbox')
                    self.driver = webdriver.Remote(
                        command_executor=f'http://{selenium_ip}:4444/wd/hub',
                        desired_capabilities=DesiredCapabilities.CHROME,
                        # options=chrome_options
                    )
                    break
                except Exception as e:
                    self.logger.info(f'faild connect to selenium. {e}')
                    time.sleep(5)
            else:
                raise Exception('faild connect to selenium.')
            if self.logger:
                self.logger.info('initialized web driver.')
            return self.driver
        except Exception as e:
            self.logger.error('error in initialization.')
            self.logger.error(e)
            if self.podname:
                delete_selenium_pod(self.podname, self.api_instance)
            raise e
            # get_dagster_logger().error(e)

    
    def stop_container(self):
        self.logger.info('finished. closing driver.')
        if self.driver:
            self.driver.quit()
            self.logger.info('closed driver.')
        if self.podname:
            delete_selenium_pod(self.podname, self.api_instance)
            # self.container.stop()            
            # self.container.remove()
            self.logger.info('closed container')


    def __enter__(self):
        self.start_container()
        return self.driver

    def __exit__(self, exception_type, exception_value, traceback):
        if self.logger:
            self.logger.info('Closing driver')
            if exception_type:
                self.logger.error(exception_type)
                self.logger.error(exception_value)
                self.logger.error(traceback)
        self.stop_container()


if __name__ == '__main__':
    with ChromeDriver() as driver:
        url = 'https://constative.com/lifestyle/snippets-of-bumble-conversations-is/3/'
        driver.get(url)
        time.sleep(3)
        page = driver.page_source
        print(page)
