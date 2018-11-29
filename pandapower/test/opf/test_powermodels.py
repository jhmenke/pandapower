# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandapower as pp
from pandapower.test.consistency_checks import consistency_checks
import numpy as np

try:
    from julia import Main
    julia_installed = True
except ImportError:
    julia_installed = False

@pytest.fixture
def net_3w_trafo_opf():

    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=220.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)
    bus5 = pp.create_bus(net, vn_kv=110.)

    pp.create_bus(net, vn_kv=110., in_service=False)

    #create 220/110 kV transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vsc_hv_percent=10., vsc_mv_percent=10.,
                                            vsc_lv_percent=10., vscr_hv_percent=0.5,
                                            vscr_mv_percent=0.5, vscr_lv_percent=0.5, pfe_mw=0.1,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_mva=100, sn_mv_mva=50, sn_lv_mva=50)

    #create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    #create loads
    pp.create_load(net, bus2, p_mw=60, controllable = False)
    pp.create_load(net, bus3, p_mw=70, controllable = False)
    pp.create_sgen(net, bus3, p_mw=10, controllable=False)

    #create generators
    pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, max_q_mvar=0.01, min_q_mvar=0)
    pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    pp.create_gen(net, bus4, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    net.gen["controllable"] = False
    return net

@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_compare_pwl_and_poly(net_3w_trafo_opf):
    net = net_3w_trafo_opf
    pp.create_pwl_cost(net, 0, 'ext_grid', [(0, 1, 1)])
    pp.create_pwl_cost(net, 0, 'gen', [(0, 30, 3), (30, 80, 3)])
    pp.create_pwl_cost(net, 1, 'gen', [(0, 100, 2)])

    pp.runpm(net)
    consistency_checks(net)

    p_gen = net.res_gen.p_mw.values
    q_gen = net.res_gen.q_mvar.values
    vm_bus = net.res_bus.vm_pu.values
    va_bus = net.res_bus.va_degree.values

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=1)
    pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=2)

    pp.runpm(net)
    consistency_checks(net)

    np.allclose(p_gen, net.res_gen.p_mw.values)
    np.allclose(q_gen, net.res_gen.q_mvar.values)
    np.allclose(vm_bus, net.res_bus.vm_pu.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)


@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_pwl():
    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus2 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus3 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)

    #create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus2, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')

    #create loads
    pp.create_load(net, bus2, p_mw=80, controllable = False)

    #create generators
    g1 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01, slack=True)
    g2 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
#    net.gen["controllable"] = False

    pp.create_pwl_cost(net, g1, 'gen', [(0, 2, 2), (2, 80, 5)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 2, 2), (2, 80, 5)])

#    pp.runpm(net)
#    consistency_checks(net, rtol=1e-3)
#    assert np.isclose(net.res_gen.p_mw.iloc[0], net.res_gen.p_mw.iloc[1])
#    assert np.isclose(net.res_gen.q_kvar.iloc[0], net.res_gen.q_kvar.iloc[1])

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    g3 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)


    pp.create_pwl_cost(net, g1, 'gen', [(0, 2, 1.), (2, 80, 8.)])
    pp.create_pwl_cost(net, g2, 'gen', [(0, 3, 2.), (3, 80, 14)])
    pp.create_pwl_cost(net, g3, 'gen', [(0, 1, 3.), (1, 80, 10.)])

    net.load.p_mw = 1
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1])

    net.load.p_mw = 3
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2]*2)

    net.load.p_mw = 5
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_gen.p_mw.at[g2], 3)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2]*2 + \
                                    net.res_gen.p_mw.at[g3]*3)

@pytest.mark.skipif(julia_installed==False, reason="requires julia installation")
def test_without_ext_grid():
    net = pp.create_empty_network()

    #create buses
    bus1 = pp.create_bus(net, vn_kv=220., geodata=(5,9))
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6,10))
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10,9))
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8,8))
    bus5 = pp.create_bus(net, vn_kv=110., geodata=(6,8))

    #create 220/110/110 kV 3W-transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vsc_hv_percent=10., vsc_mv_percent=10.,
                                            vsc_lv_percent=10., vscr_hv_percent=0.5,
                                            vscr_mv_percent=0.5, vscr_lv_percent=0.5, pfe_mw=0.1,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_mva=100, sn_mv_mva=50, sn_lv_mva=50)

    #create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    #create loads
    pp.create_load(net, bus2, p_mw=60)
    pp.create_load(net, bus3, p_mw=70)
    pp.create_load(net, bus4, p_mw=10)

    #create generators
    g1 = pp.create_gen(net, bus1, p_mw=40, min_p_mw=0, max_q_mvar=0.020, slack=True)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=1000)

    g2 = pp.create_gen(net, bus3, p_mw=40, min_p_mw=0, max_q_mvar=0.020, vm_pu=1.01)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=2000)

    g3 = pp.create_gen(net, bus4, p_mw=0.050, min_p_mw=0, max_q_mvar=0.020, vm_pu=1.01)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=3000)

    pp.runpm(net)

    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1]*1e3)

    net.trafo3w["max_loading_percent"] = 50
    pp.runpm(net)
    consistency_checks(net, rtol=1e-3)
    assert 49.9 < net.res_trafo3w.loading_percent.values[0] < 50
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1]*1e3 + net.res_gen.p_mw.at[g2]*2e3)

if __name__ == '__main__':
    pytest.main([])