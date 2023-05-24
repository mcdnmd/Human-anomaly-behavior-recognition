from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = ROOT / 'data'


def main():
    video = []
    action = []
    start_frame = []
    end_frame = []
    start_frame2 = []
    end_frame2 = []
    with open(str(DATA_ROOT / "annotation_for_ucf.txt")) as file:
        line = file.readline().replace('\n', '')
        while line != "":
            data = line.split('  ')
            video.append(data[0])
            action.append(data[1].lower())
            start_frame.append(int(data[2]))
            end_frame.append(int(data[3]))
            start_frame2.append(int(data[4]))
            end_frame2.append(int(data[5]))
            line = file.readline().replace('\n', '')
    df = pd.DataFrame()
    df['video'] = video
    df['action'] = action
    df['start_frame'] = start_frame
    df['end_frame'] = end_frame
    df['start_frame2'] = start_frame2
    df['end_frame2'] = end_frame2
    df.to_csv(str(DATA_ROOT / "annotation_for_ucf.csv"), sep=';', index=False)


if __name__ == '__main__':
    main()