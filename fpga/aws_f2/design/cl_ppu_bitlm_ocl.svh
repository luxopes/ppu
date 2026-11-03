// =============================================================================
// Vlozka do CL_TEMPLATE.sv (aws-fpga, vetev f2) pro CL "cl_ppu_bitlm".
// Nahrazuje tie-off OCL rozhrani (cl_ocl_awready = 'b0; ... cl_ocl_rdata = 'b0;)
// instanci BitLM jadra. Nazvy portu dle hdk/common/shell_stable/design/
// interfaces/cl_ports.vh (f2): ocl_cl_* vstupy, cl_ocl_* vystupy,
// hodiny clk_main_a0 (250 MHz), reset rst_main_n (synchronni, aktivni v 0).
// Postup viz fpga/aws_f2/README.md.
// =============================================================================
    logic bitlm_busy;

    bitlm_fpga_top #(
        .LOG2_N(3), .LOG2_E(5), .LOG2_H(9), .LOG2_L(6)
    ) u_bitlm (
        .clk            (clk_main_a0),
        .rst_n          (rst_main_n),
        .s_axi_awaddr   (ocl_cl_awaddr),
        .s_axi_awvalid  (ocl_cl_awvalid),
        .s_axi_awready  (cl_ocl_awready),
        .s_axi_wdata    (ocl_cl_wdata),
        .s_axi_wstrb    (ocl_cl_wstrb),
        .s_axi_wvalid   (ocl_cl_wvalid),
        .s_axi_wready   (cl_ocl_wready),
        .s_axi_bresp    (cl_ocl_bresp),
        .s_axi_bvalid   (cl_ocl_bvalid),
        .s_axi_bready   (ocl_cl_bready),
        .s_axi_araddr   (ocl_cl_araddr),
        .s_axi_arvalid  (ocl_cl_arvalid),
        .s_axi_arready  (cl_ocl_arready),
        .s_axi_rdata    (cl_ocl_rdata),
        .s_axi_rresp    (cl_ocl_rresp),
        .s_axi_rvalid   (cl_ocl_rvalid),
        .s_axi_rready   (ocl_cl_rready),
        .busy           (bitlm_busy)
    );
