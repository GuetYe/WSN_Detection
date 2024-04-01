import argparse   #步骤一
from processed.IBRL_load import get_IBRL_data

def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser(description="you should add those parameter")        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    parser.add_argument('--gpu', default=True, help="if you want to use cuda (True)")
    parser.add_argument('--dataset', default="IBRL", help="dataset name")
    parser.add_argument('--window_size', default=20, help="slid window length")
    parser.add_argument('--batch_size', default=4, help="training batch size")
    parser.add_argument('--test_time', default=100, help="number of test times")
    parser.add_argument('--margin', default=20, help="the margin of test check point")
    parser.add_argument('--inject_length', default=5, help="inject_length")
    parser.add_argument('--train_rate', default=0.6, help="train_rate")
    parser.add_argument('--epoch', default=150, help="train_rate")

    args = parser.parse_args()
    return args

def get_data_inf(datasetName):
    if datasetName == "IBRL":
        IBRL_data = get_IBRL_data()
        node_num = IBRL_data.shape[1]
        mode_num = IBRL_data.shape[0]
        return node_num, mode_num

def get_inject_inf(datasetName):
    IBRL_inject_point = [
                         [50, 149, 263, 364, 452,
                          550, 645, 761, 853, 944,
                          1049, 1152, 1262, 1358, 1462,
                          1566, 1659, 1768, 1847, 1951, ],
                        [53, 144, 265, 360, 453,
                         557, 641, 760, 856, 942,
                         1043, 1156, 1258, 1352, 1468,
                         1561, 1652, 1763, 1844, 1959, ],
                        [33, 138, 261, 362, 451,
                         537, 640, 765, 859, 946,
                         1073, 1167, 1270, 1332, 1462,
                         1551, 1653, 1752, 1874, 1951, ],
                        [442, 548, 665, 767, 846,
                         939, 1047, 1163, 1252, 1346,
                         1470, 1560, 1650, 1744, 1860,
                         1950, 2050, 2154, 2263, 2357, ]
                        ]
    IBRL_inject_mode = [[0, 0, 1, 0, 1,
                         0, 1, 1, 0, 1,
                         0, 2, 2, 2, 2,
                         0, 1, 2, 1, 2, ],
                       [1, 2, 2, 0, 1,
                        0, 1, 2, 0, 1,
                        0, 2, 0, 0, 2,
                        0, 1, 0, 1, 2, ],
                        [1, 2, 2, 0, 1,
                         1, 2, 2, 2, 1,
                         0, 0, 0, 1, 0,
                         1, 1, 1, 1, 2, ],
                        [1, 2, 2, 0, 1,
                         0, 1, 2, 0, 1,
                         0, 2, 0, 0, 2,
                         0, 1, 0, 1, 2, ],
                        ]
    IBRL_inject_node = [[i for i in range(20)],[i+20 for i in range(20)],[i+31 for i in range(20)],[i+20 for i in range(20)]]

    # CIMIS_inject_point = [[35, 151, 252, 353, 454, 555, 656, 757],
    #                      [78, 153, 245, 344, 468, 542, 649, 770],
    #                      [55, 139, 262, 323, 469, 560, 638, 747],
    #                      [57, 164, 253, 349, 429, 571, 668, 754],
    #                      [50, 169, 256, 341, 477, 533, 640, 772],
    #                      [66, 172, 238, 350, 444, 555, 642, 764],
    #                      [78, 153, 245, 344, 468, 542, 649, 770]]
    # CIMIS_inject_mode = [[0, 1, 2, 3, 4, 5, 6, 7],
    #                      [1, 3, 0, 2, 4, 7, 6, 5],
    #                      [7, 6, 5, 3, 4, 1, 0, 2],
    #                      [5, 7, 1, 0, 3, 2, 6, 4],
    #                      [7, 1, 5, 0, 2, 3, 4, 6],
    #                      [1, 0, 3, 2, 7, 4, 6, 5],
    #                      [0, 1, 2, 3, 4, 5, 6, 7]]
    # CIMIS_inject_node = [[i for i in range(8)],
    #                      [i+8 for i in range(8)],
    #                      [i+16 for i in range(8)],
    #                      [i+24 for i in range(8)],
    #                      [i+32 for i in range(8)],
    #                      [i+40 for i in range(8)],
    #                      [i+49 for i in range(8)],]
    if datasetName == "IBRL":
        return IBRL_inject_point,IBRL_inject_mode,IBRL_inject_node
    # elif datasetName =="CIMIS":
    #     return CIMIS_inject_point,CIMIS_inject_mode,CIMIS_inject_node