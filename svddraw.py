import fwmodel
import draw

draw.nohub_draw_cycle()
se, ue, si, ui, c, m = fwmodel.load_compact_svd('C:\\dev\\3dface\\svd2',40,25)
draw.write_parameters(0,fwmodel.restore_vector(c,ui[4],ue[0]) + m)

