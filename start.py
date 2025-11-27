from selfmotionestimation.pipeline.pipeline import Pipeline
from selfmotionestimation.data.log.logger import Logger

LOG = Logger("START")

# Pipeline starter

def main():
    LOG.info("Pipeline starting...")

    Pipeline(calibration=False,
             png_dump=True,
             processing=True,
             visualisation=True,
             comparing=True)

    LOG.info("Pipeline ran.")

if __name__ == "__main__":
    main()