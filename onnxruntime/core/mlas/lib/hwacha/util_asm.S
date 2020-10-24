    .text
    .align 10
    .globl vtest_avi
vtest_avi: 
    vpset vp0
    vstop
    
    .globl vsetvlen
vsetvlen:
    vpset vp0
    vstop

    .globl vtest2
vtest2:
    vpset vp0
    vlb vv0, va1 #input
    vlb vv1, va2 #filter
    vlb vv2, va0 #output
    vmul vv0, vv0, vv1
    vadd vv0, vv0, vv2
    vdiv vv0, vv0, vs1 
    vsb vv0, va0
    vstop

    .globl vtest3
vtest3:
    vpset vp0
    vlb vv0, va1 #input
    vlb vv1, va2 #filter
    vlb vv2, va0 #output
    vsb vv0, va0
    vstop

    .globl vtest4
vtest4:
    vpset vp0
    vlb vv0, va1 #input
    vlb vv1, va2 #filter
    vlh vv2, va0 #temp_output
    vmul vv0, vv0, vv1
    vadd vv0, vv0, vv2
    vsh vv0, va0
    vstop

    .globl vtest5
vtest5:
    vpset vp0
    vlh vv0, va0 #temp_output
    vdiv vv0, vv0, vs1
    vsb vv0, va1
    vstop

    .globl accumulate_channels
accumulate_channels:
    vpset vp0
    vlsegstb vv0, va0, va3, 0 #first accumulate_channel
    vlsegstb vv1, va1, va3, 0 #second accumulate_channels
    vadd vv0, vv0, vv1
    vsb vv0, va2 #output 
    vstop



    