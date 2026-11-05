// ============================================================================
// Amazon FPGA Hardware Development Kit
//
// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Amazon Software License (the "License"). You may not use
// this file except in compliance with the License. A copy of the License is
// located at
//
//    http://aws.amazon.com/asl/
//
// or in the "license" file accompanying this file. This file is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
// implied. See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================


//====================================================================================
// Top level module file for cl_bittf
// BitTF (PPU): 1-bitovy jazykovy model trenovany na FPGA. Jadro bittf_core2 s registry na OCL (BAR0)
// a pameti v DDR; host do DDR pise/cte pres PCIS (BAR4, DDR od adresy 0), kdyz jadro nebezi.
//====================================================================================

module cl_bittf
    #(
      parameter EN_DDR = 1,
      parameter EN_HBM = 0
    )
    (
      `include "cl_ports.vh"
    );

`include "cl_id_defines.vh" // CL ID defines required for all examples
`include "cl_bittf_defines.vh"


//=============================================================================
// GLOBALS
//=============================================================================

  always_comb begin
     cl_sh_status0     = 'b0;
     cl_sh_status1     = 'b0;
     cl_sh_status2     = 'b0;
     cl_sh_id0         = `CL_SH_ID0;
     cl_sh_id1         = `CL_SH_ID1;
     cl_sh_status_vled = 'b0;
     cl_sh_dma_wr_full = 'b0;
     cl_sh_dma_rd_full = 'b0;
  end


//=============================================================================
// PCIM
//=============================================================================

  // Cause Protocol Violations
  always_comb begin
    cl_sh_pcim_awaddr  = 'b0;
    cl_sh_pcim_awsize  = 'b0;
    cl_sh_pcim_awburst = 'b0;
    cl_sh_pcim_awvalid = 'b0;

    cl_sh_pcim_wdata   = 'b0;
    cl_sh_pcim_wstrb   = 'b0;
    cl_sh_pcim_wlast   = 'b0;
    cl_sh_pcim_wvalid  = 'b0;

    cl_sh_pcim_araddr  = 'b0;
    cl_sh_pcim_arsize  = 'b0;
    cl_sh_pcim_arburst = 'b0;
    cl_sh_pcim_arvalid = 'b0;
  end

  // Remaining CL Output Ports
  always_comb begin
    cl_sh_pcim_awid    = 'b0;
    cl_sh_pcim_awlen   = 'b0;
    cl_sh_pcim_awcache = 'b0;
    cl_sh_pcim_awlock  = 'b0;
    cl_sh_pcim_awprot  = 'b0;
    cl_sh_pcim_awqos   = 'b0;
    cl_sh_pcim_awuser  = 'b0;

    cl_sh_pcim_wid     = 'b0;
    cl_sh_pcim_wuser   = 'b0;

    cl_sh_pcim_arid    = 'b0;
    cl_sh_pcim_arlen   = 'b0;
    cl_sh_pcim_arcache = 'b0;
    cl_sh_pcim_arlock  = 'b0;
    cl_sh_pcim_arprot  = 'b0;
    cl_sh_pcim_arqos   = 'b0;
    cl_sh_pcim_aruser  = 'b0;

    cl_sh_pcim_rready  = 'b0;
  end

//=============================================================================
// SDA
//=============================================================================

  // Cause Protocol Violations
  always_comb begin
    cl_sda_bresp   = 'b0;
    cl_sda_rresp   = 'b0;
    cl_sda_rvalid  = 'b0;
  end

  // Remaining CL Output Ports
  always_comb begin
    cl_sda_awready = 'b0;
    cl_sda_wready  = 'b0;

    cl_sda_bvalid = 'b0;

    cl_sda_arready = 'b0;

    cl_sda_rdata   = 'b0;
  end

//=============================================================================
// BitTF: reset, FLR
//=============================================================================

  logic rst_q1, rst_q2;
  always_ff @(posedge clk_main_a0) begin
    rst_q1 <= rst_main_n;
    rst_q2 <= rst_q1;
  end
  wire bt_rst_n = rst_q2;

  logic flr_q;
  always_ff @(posedge clk_main_a0)
    if (!bt_rst_n) begin
      flr_q          <= 1'b0;
      cl_sh_flr_done <= 1'b0;
    end else begin
      flr_q          <= sh_cl_flr_assert;
      cl_sh_flr_done <= flr_q && !cl_sh_flr_done;
    end

//=============================================================================
// BitTF: hodiny jadra (MMCM z clk_main_a0) a reset v jejich domene
//=============================================================================

  // Jadro bezi na vlastnich hodinach clk_core = 1000 MHz / CORE_DIV (VCO 1 GHz z 250 MHz).
  // Registry (OCL) a pamet (AXI4 do prepinace na clk_main_a0) prechazi pres asynchronni prevodniky.
  localparam real CORE_DIV = `BT_CORE_DIV;

  wire clk_core_mmcm, clk_core, mmcm_fb, mmcm_fb_buf, mmcm_locked;
  MMCME4_BASE #(
    .CLKIN1_PERIOD    (4.000),
    .CLKFBOUT_MULT_F  (4.000),
    .DIVCLK_DIVIDE    (1),
    .CLKOUT0_DIVIDE_F (CORE_DIV)
  ) CORE_MMCM (
    .CLKIN1   (clk_main_a0),
    .CLKFBIN  (mmcm_fb_buf),
    .CLKFBOUT (mmcm_fb),
    .CLKOUT0  (clk_core_mmcm),
    .LOCKED   (mmcm_locked),
    .RST      (!bt_rst_n),
    .PWRDWN   (1'b0),
    .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
    .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6()
  );
  BUFG FB_BUF   (.I(mmcm_fb),       .O(mmcm_fb_buf));
  BUFG CORE_BUF (.I(clk_core_mmcm), .O(clk_core));

  (* ASYNC_REG = "TRUE" *) logic crst_q1, crst_q2;
  always_ff @(posedge clk_core or negedge mmcm_locked)
    if (!mmcm_locked) begin
      crst_q1 <= 1'b0;
      crst_q2 <= 1'b0;
    end else begin
      crst_q1 <= bt_rst_n;
      crst_q2 <= crst_q1;
    end
  wire core_rst_n = crst_q2;

  // stav pro hosta / prepinac: synchronizace mezi domenami (pomale urovnove signaly)
  wire          bt_busy;                 // clk_core
  wire          own_core;                // clk_main_a0
  logic         ddr_ready;               // clk_main_a0
  (* ASYNC_REG = "TRUE" *) logic busy_m1, busy_m2;          // -> clk_main_a0
  (* ASYNC_REG = "TRUE" *) logic [1:0] st_c1, st_c2;        // -> clk_core
  always_ff @(posedge clk_main_a0) begin
    busy_m1 <= bt_busy;
    busy_m2 <= busy_m1;
  end
  always_ff @(posedge clk_core) begin
    st_c1 <= {own_core, ddr_ready};
    st_c2 <= st_c1;
  end

//=============================================================================
// BitTF: jadro (registry pres OCL, pamet pres AXI4) a prepinac pameti host/jadro
//=============================================================================

  // OCL (clk_main_a0) -> registry jadra (clk_core)
  wire [31:0]   r_awaddr, r_wdata, r_araddr, r_rdata;
  wire [3:0]    r_wstrb;
  wire [1:0]    r_bresp, r_rresp;
  wire          r_awvalid, r_awready, r_wvalid, r_wready, r_bvalid, r_bready;
  wire          r_arvalid, r_arready, r_rvalid, r_rready;

  cl_axi_clock_converter_light OCL_CDC (
    .s_axi_aclk    (clk_main_a0),
    .s_axi_aresetn (bt_rst_n),
    .s_axi_awaddr  (ocl_cl_awaddr),
    .s_axi_awprot  (3'b0),
    .s_axi_awvalid (ocl_cl_awvalid),
    .s_axi_awready (cl_ocl_awready),
    .s_axi_wdata   (ocl_cl_wdata),
    .s_axi_wstrb   (ocl_cl_wstrb),
    .s_axi_wvalid  (ocl_cl_wvalid),
    .s_axi_wready  (cl_ocl_wready),
    .s_axi_bresp   (cl_ocl_bresp),
    .s_axi_bvalid  (cl_ocl_bvalid),
    .s_axi_bready  (ocl_cl_bready),
    .s_axi_araddr  (ocl_cl_araddr),
    .s_axi_arprot  (3'b0),
    .s_axi_arvalid (ocl_cl_arvalid),
    .s_axi_arready (cl_ocl_arready),
    .s_axi_rdata   (cl_ocl_rdata),
    .s_axi_rresp   (cl_ocl_rresp),
    .s_axi_rvalid  (cl_ocl_rvalid),
    .s_axi_rready  (ocl_cl_rready),
    .m_axi_aclk    (clk_core),
    .m_axi_aresetn (core_rst_n),
    .m_axi_awaddr  (r_awaddr),
    .m_axi_awprot  (),
    .m_axi_awvalid (r_awvalid),
    .m_axi_awready (r_awready),
    .m_axi_wdata   (r_wdata),
    .m_axi_wstrb   (r_wstrb),
    .m_axi_wvalid  (r_wvalid),
    .m_axi_wready  (r_wready),
    .m_axi_bresp   (r_bresp),
    .m_axi_bvalid  (r_bvalid),
    .m_axi_bready  (r_bready),
    .m_axi_araddr  (r_araddr),
    .m_axi_arprot  (),
    .m_axi_arvalid (r_arvalid),
    .m_axi_arready (r_arready),
    .m_axi_rdata   (r_rdata),
    .m_axi_rresp   (r_rresp),
    .m_axi_rvalid  (r_rvalid),
    .m_axi_rready  (r_rready)
  );

  // AXI4 jadra v domene clk_core (jednobeatove, ID 0)
  wire [63:0]   k_araddr, k_awaddr;
  wire [7:0]    k_arlen, k_awlen;
  wire [2:0]    k_arsize, k_awsize;
  wire [1:0]    k_arburst, k_awburst;
  wire          k_arvalid, k_arready, k_rvalid, k_rready;
  wire          k_awvalid, k_awready, k_wvalid, k_wready, k_wlast, k_bvalid, k_bready;
  wire [511:0]  k_rdata, k_wdata;
  wire [63:0]   k_wstrb;

  bittf_fpga_top #(.LB(64), .TM(32), .TN(32)) BITTF (
    .clk            (clk_core),
    .rst_n          (core_rst_n),
    .s_axi_awaddr   (r_awaddr),
    .s_axi_awvalid  (r_awvalid),
    .s_axi_awready  (r_awready),
    .s_axi_wdata    (r_wdata),
    .s_axi_wstrb    (r_wstrb),
    .s_axi_wvalid   (r_wvalid),
    .s_axi_wready   (r_wready),
    .s_axi_bresp    (r_bresp),
    .s_axi_bvalid   (r_bvalid),
    .s_axi_bready   (r_bready),
    .s_axi_araddr   (r_araddr),
    .s_axi_arvalid  (r_arvalid),
    .s_axi_arready  (r_arready),
    .s_axi_rdata    (r_rdata),
    .s_axi_rresp    (r_rresp),
    .s_axi_rvalid   (r_rvalid),
    .s_axi_rready   (r_rready),
    .m_axi_araddr   (k_araddr),
    .m_axi_arlen    (k_arlen),
    .m_axi_arsize   (k_arsize),
    .m_axi_arburst  (k_arburst),
    .m_axi_arvalid  (k_arvalid),
    .m_axi_arready  (k_arready),
    .m_axi_rdata    (k_rdata),
    .m_axi_rvalid   (k_rvalid),
    .m_axi_rready   (k_rready),
    .m_axi_awaddr   (k_awaddr),
    .m_axi_awlen    (k_awlen),
    .m_axi_awsize   (k_awsize),
    .m_axi_awburst  (k_awburst),
    .m_axi_awvalid  (k_awvalid),
    .m_axi_awready  (k_awready),
    .m_axi_wdata    (k_wdata),
    .m_axi_wstrb    (k_wstrb),
    .m_axi_wlast    (k_wlast),
    .m_axi_wvalid   (k_wvalid),
    .m_axi_wready   (k_wready),
    .m_axi_bvalid   (k_bvalid),
    .m_axi_bready   (k_bready),
    .busy           (bt_busy),
    .ext_status     ({30'd0, st_c2})
  );

  // pamet jadra (clk_core) -> prepinac (clk_main_a0)
  wire [63:0]   c_araddr, c_awaddr;
  wire [7:0]    c_arlen, c_awlen;
  wire [2:0]    c_arsize, c_awsize;
  wire [1:0]    c_arburst, c_awburst;
  wire          c_arvalid, c_arready, c_rvalid, c_rready;
  wire          c_awvalid, c_awready, c_wvalid, c_wready, c_wlast, c_bvalid, c_bready;
  wire [511:0]  c_rdata, c_wdata;
  wire [63:0]   c_wstrb;

  cl_axi_clock_converter MEM_CDC (
    .s_axi_aclk     (clk_core),
    .s_axi_aresetn  (core_rst_n),
    .s_axi_awid     (16'd0),
    .s_axi_awaddr   (k_awaddr),
    .s_axi_awlen    (k_awlen),
    .s_axi_awsize   (k_awsize),
    .s_axi_awburst  (k_awburst),
    .s_axi_awlock   (1'b0),
    .s_axi_awcache  (4'b0011),
    .s_axi_awprot   (3'b0),
    .s_axi_awregion (4'b0),
    .s_axi_awqos    (4'b0),
    .s_axi_awuser   (19'd0),
    .s_axi_awvalid  (k_awvalid),
    .s_axi_awready  (k_awready),
    .s_axi_wdata    (k_wdata),
    .s_axi_wstrb    (k_wstrb),
    .s_axi_wlast    (k_wlast),
    .s_axi_wvalid   (k_wvalid),
    .s_axi_wready   (k_wready),
    .s_axi_bid      (),
    .s_axi_bresp    (),
    .s_axi_bvalid   (k_bvalid),
    .s_axi_bready   (k_bready),
    .s_axi_arid     (16'd0),
    .s_axi_araddr   (k_araddr),
    .s_axi_arlen    (k_arlen),
    .s_axi_arsize   (k_arsize),
    .s_axi_arburst  (k_arburst),
    .s_axi_arlock   (1'b0),
    .s_axi_arcache  (4'b0011),
    .s_axi_arprot   (3'b0),
    .s_axi_arregion (4'b0),
    .s_axi_arqos    (4'b0),
    .s_axi_aruser   (19'd0),
    .s_axi_arvalid  (k_arvalid),
    .s_axi_arready  (k_arready),
    .s_axi_rid      (),
    .s_axi_rdata    (k_rdata),
    .s_axi_rresp    (),
    .s_axi_rlast    (),
    .s_axi_rvalid   (k_rvalid),
    .s_axi_rready   (k_rready),
    .m_axi_aclk     (clk_main_a0),
    .m_axi_aresetn  (bt_rst_n),
    .m_axi_awid     (),
    .m_axi_awaddr   (c_awaddr),
    .m_axi_awlen    (c_awlen),
    .m_axi_awsize   (c_awsize),
    .m_axi_awburst  (c_awburst),
    .m_axi_awlock   (),
    .m_axi_awcache  (),
    .m_axi_awprot   (),
    .m_axi_awregion (),
    .m_axi_awqos    (),
    .m_axi_awuser   (),
    .m_axi_awvalid  (c_awvalid),
    .m_axi_awready  (c_awready),
    .m_axi_wdata    (c_wdata),
    .m_axi_wstrb    (c_wstrb),
    .m_axi_wlast    (c_wlast),
    .m_axi_wvalid   (c_wvalid),
    .m_axi_wready   (c_wready),
    .m_axi_bid      (16'd0),
    .m_axi_bresp    (2'b00),
    .m_axi_bvalid   (c_bvalid),
    .m_axi_bready   (c_bready),
    .m_axi_arid     (),
    .m_axi_araddr   (c_araddr),
    .m_axi_arlen    (c_arlen),
    .m_axi_arsize   (c_arsize),
    .m_axi_arburst  (c_arburst),
    .m_axi_arlock   (),
    .m_axi_arcache  (),
    .m_axi_arprot   (),
    .m_axi_arregion (),
    .m_axi_arqos    (),
    .m_axi_aruser   (),
    .m_axi_arvalid  (c_arvalid),
    .m_axi_arready  (c_arready),
    .m_axi_rid      (16'd0),
    .m_axi_rdata    (c_rdata),
    .m_axi_rresp    (2'b00),
    .m_axi_rlast    (1'b1),
    .m_axi_rvalid   (c_rvalid),
    .m_axi_rready   (c_rready)
  );

  // pamet DDR (AXI4 512b) sdilena hostem (PCIS, BAR4 od adresy 0) a jadrem
  wire [15:0]   d_awid, d_bid, d_arid, d_rid;
  wire [63:0]   d_awaddr, d_araddr;
  wire [7:0]    d_awlen, d_arlen;
  wire [2:0]    d_awsize, d_arsize;
  wire [1:0]    d_awburst, d_arburst, d_bresp, d_rresp;
  wire          d_awvalid, d_awready, d_wlast, d_wvalid, d_wready, d_bvalid, d_bready;
  wire          d_arvalid, d_arready, d_rlast, d_rvalid, d_rready;
  wire [511:0]  d_wdata, d_rdata;
  wire [63:0]   d_wstrb;

  axi_own_mux #(.DW(512), .IW(16)) DDR_MUX (
    .clk        (clk_main_a0),
    .rst_n      (bt_rst_n),
    .core_req   (busy_m2),
    .own_core   (own_core),
    .h_awid     (sh_cl_dma_pcis_awid),
    .h_awaddr   (sh_cl_dma_pcis_awaddr),
    .h_awlen    (sh_cl_dma_pcis_awlen),
    .h_awsize   (sh_cl_dma_pcis_awsize),
    .h_awburst  (sh_cl_dma_pcis_awburst),
    .h_awvalid  (sh_cl_dma_pcis_awvalid),
    .h_awready  (cl_sh_dma_pcis_awready),
    .h_wdata    (sh_cl_dma_pcis_wdata),
    .h_wstrb    (sh_cl_dma_pcis_wstrb),
    .h_wlast    (sh_cl_dma_pcis_wlast),
    .h_wvalid   (sh_cl_dma_pcis_wvalid),
    .h_wready   (cl_sh_dma_pcis_wready),
    .h_bid      (cl_sh_dma_pcis_bid),
    .h_bresp    (cl_sh_dma_pcis_bresp),
    .h_bvalid   (cl_sh_dma_pcis_bvalid),
    .h_bready   (sh_cl_dma_pcis_bready),
    .h_arid     (sh_cl_dma_pcis_arid),
    .h_araddr   (sh_cl_dma_pcis_araddr),
    .h_arlen    (sh_cl_dma_pcis_arlen),
    .h_arsize   (sh_cl_dma_pcis_arsize),
    .h_arburst  (sh_cl_dma_pcis_arburst),
    .h_arvalid  (sh_cl_dma_pcis_arvalid),
    .h_arready  (cl_sh_dma_pcis_arready),
    .h_rid      (cl_sh_dma_pcis_rid),
    .h_rdata    (cl_sh_dma_pcis_rdata),
    .h_rresp    (cl_sh_dma_pcis_rresp),
    .h_rlast    (cl_sh_dma_pcis_rlast),
    .h_rvalid   (cl_sh_dma_pcis_rvalid),
    .h_rready   (sh_cl_dma_pcis_rready),
    .c_awaddr   (c_awaddr),
    .c_awlen    (c_awlen),
    .c_awsize   (c_awsize),
    .c_awburst  (c_awburst),
    .c_awvalid  (c_awvalid),
    .c_awready  (c_awready),
    .c_wdata    (c_wdata),
    .c_wstrb    (c_wstrb),
    .c_wlast    (c_wlast),
    .c_wvalid   (c_wvalid),
    .c_wready   (c_wready),
    .c_bvalid   (c_bvalid),
    .c_bready   (c_bready),
    .c_araddr   (c_araddr),
    .c_arlen    (c_arlen),
    .c_arsize   (c_arsize),
    .c_arburst  (c_arburst),
    .c_arvalid  (c_arvalid),
    .c_arready  (c_arready),
    .c_rdata    (c_rdata),
    .c_rvalid   (c_rvalid),
    .c_rready   (c_rready),
    .m_awid     (d_awid),
    .m_awaddr   (d_awaddr),
    .m_awlen    (d_awlen),
    .m_awsize   (d_awsize),
    .m_awburst  (d_awburst),
    .m_awvalid  (d_awvalid),
    .m_awready  (d_awready),
    .m_wdata    (d_wdata),
    .m_wstrb    (d_wstrb),
    .m_wlast    (d_wlast),
    .m_wvalid   (d_wvalid),
    .m_wready   (d_wready),
    .m_bid      (d_bid),
    .m_bresp    (d_bresp),
    .m_bvalid   (d_bvalid),
    .m_bready   (d_bready),
    .m_arid     (d_arid),
    .m_araddr   (d_araddr),
    .m_arlen    (d_arlen),
    .m_arsize   (d_arsize),
    .m_arburst  (d_arburst),
    .m_arvalid  (d_arvalid),
    .m_arready  (d_arready),
    .m_rid      (d_rid),
    .m_rdata    (d_rdata),
    .m_rresp    (d_rresp),
    .m_rlast    (d_rlast),
    .m_rvalid   (d_rvalid),
    .m_rready   (d_rready)
  );

  always_comb cl_sh_dma_pcis_ruser = 'b0;

//=============================================================================
// SH_DDR
//=============================================================================

  logic         stat_wr_q, stat_rd_q, stat_ack_q;
  logic [7:0]   stat_int_q;
  logic [7:0]   stat_addr_q;
  logic [31:0]  stat_wdata_q, stat_rdata_q;

  lib_pipe #(.WIDTH(1+1+8+32), .STAGES(1)) PIPE_DDR_STAT0 (
    .clk     (clk_main_a0),
    .rst_n   (bt_rst_n),
    .in_bus  ({sh_cl_ddr_stat_wr, sh_cl_ddr_stat_rd, sh_cl_ddr_stat_addr, sh_cl_ddr_stat_wdata}),
    .out_bus ({stat_wr_q, stat_rd_q, stat_addr_q, stat_wdata_q})
  );

  lib_pipe #(.WIDTH(1+8+32), .STAGES(1)) PIPE_DDR_STAT_ACK0 (
    .clk     (clk_main_a0),
    .rst_n   (bt_rst_n),
    .in_bus  ({stat_ack_q, stat_int_q, stat_rdata_q}),
    .out_bus ({cl_sh_ddr_stat_ack, cl_sh_ddr_stat_int, cl_sh_ddr_stat_rdata})
  );

   sh_ddr
     #(
       .DDR_PRESENT (EN_DDR)
       )
   SH_DDR
     (
      .clk                       (clk_main_a0 ),
      .rst_n                     (bt_rst_n    ),
      .stat_clk                  (clk_main_a0 ),
      .stat_rst_n                (bt_rst_n    ),
      .CLK_DIMM_DP               (CLK_DIMM_DP ),
      .CLK_DIMM_DN               (CLK_DIMM_DN ),
      .M_ACT_N                   (M_ACT_N     ),
      .M_MA                      (M_MA        ),
      .M_BA                      (M_BA        ),
      .M_BG                      (M_BG        ),
      .M_CKE                     (M_CKE       ),
      .M_ODT                     (M_ODT       ),
      .M_CS_N                    (M_CS_N      ),
      .M_CLK_DN                  (M_CLK_DN    ),
      .M_CLK_DP                  (M_CLK_DP    ),
      .M_PAR                     (M_PAR       ),
      .M_DQ                      (M_DQ        ),
      .M_ECC                     (M_ECC       ),
      .M_DQS_DP                  (M_DQS_DP    ),
      .M_DQS_DN                  (M_DQS_DN    ),
      .cl_RST_DIMM_N             (RST_DIMM_N  ),
      .cl_sh_ddr_axi_awid        (d_awid      ),
      .cl_sh_ddr_axi_awaddr      (d_awaddr    ),
      .cl_sh_ddr_axi_awlen       (d_awlen     ),
      .cl_sh_ddr_axi_awsize      (d_awsize    ),
      .cl_sh_ddr_axi_awvalid     (d_awvalid   ),
      .cl_sh_ddr_axi_awburst     (d_awburst   ),
      .cl_sh_ddr_axi_awuser      (1'b0        ),
      .cl_sh_ddr_axi_awready     (d_awready   ),
      .cl_sh_ddr_axi_wdata       (d_wdata     ),
      .cl_sh_ddr_axi_wstrb       (d_wstrb     ),
      .cl_sh_ddr_axi_wlast       (d_wlast     ),
      .cl_sh_ddr_axi_wvalid      (d_wvalid    ),
      .cl_sh_ddr_axi_wready      (d_wready    ),
      .cl_sh_ddr_axi_bid         (d_bid       ),
      .cl_sh_ddr_axi_bresp       (d_bresp     ),
      .cl_sh_ddr_axi_bvalid      (d_bvalid    ),
      .cl_sh_ddr_axi_bready      (d_bready    ),
      .cl_sh_ddr_axi_arid        (d_arid      ),
      .cl_sh_ddr_axi_araddr      (d_araddr    ),
      .cl_sh_ddr_axi_arlen       (d_arlen     ),
      .cl_sh_ddr_axi_arsize      (d_arsize    ),
      .cl_sh_ddr_axi_arvalid     (d_arvalid   ),
      .cl_sh_ddr_axi_arburst     (d_arburst   ),
      .cl_sh_ddr_axi_aruser      (1'b0        ),
      .cl_sh_ddr_axi_arready     (d_arready   ),
      .cl_sh_ddr_axi_rid         (d_rid       ),
      .cl_sh_ddr_axi_rdata       (d_rdata     ),
      .cl_sh_ddr_axi_rresp       (d_rresp     ),
      .cl_sh_ddr_axi_rlast       (d_rlast     ),
      .cl_sh_ddr_axi_rvalid      (d_rvalid    ),
      .cl_sh_ddr_axi_rready      (d_rready    ),
      .sh_ddr_stat_bus_addr      (stat_addr_q ),
      .sh_ddr_stat_bus_wdata     (stat_wdata_q),
      .sh_ddr_stat_bus_wr        (stat_wr_q   ),
      .sh_ddr_stat_bus_rd        (stat_rd_q   ),
      .sh_ddr_stat_bus_ack       (stat_ack_q  ),
      .sh_ddr_stat_bus_rdata     (stat_rdata_q),
      .ddr_sh_stat_int           (stat_int_q  ),
      .sh_cl_ddr_is_ready        (ddr_ready   )
      );

//=============================================================================
// USER-DEFIEND INTERRUPTS
//=============================================================================

  always_comb begin
    cl_sh_apppf_irq_req = 'b0;
  end

//=============================================================================
// VIRTUAL JTAG
//=============================================================================

  always_comb begin
    tdo = 'b0;
  end

//=============================================================================
// HBM MONITOR IO
//=============================================================================

  always_comb begin
    hbm_apb_paddr_1   = 'b0;
    hbm_apb_pprot_1   = 'b0;
    hbm_apb_psel_1    = 'b0;
    hbm_apb_penable_1 = 'b0;
    hbm_apb_pwrite_1  = 'b0;
    hbm_apb_pwdata_1  = 'b0;
    hbm_apb_pstrb_1   = 'b0;
    hbm_apb_pready_1  = 'b0;
    hbm_apb_prdata_1  = 'b0;
    hbm_apb_pslverr_1 = 'b0;

    hbm_apb_paddr_0   = 'b0;
    hbm_apb_pprot_0   = 'b0;
    hbm_apb_psel_0    = 'b0;
    hbm_apb_penable_0 = 'b0;
    hbm_apb_pwrite_0  = 'b0;
    hbm_apb_pwdata_0  = 'b0;
    hbm_apb_pstrb_0   = 'b0;
    hbm_apb_pready_0  = 'b0;
    hbm_apb_prdata_0  = 'b0;
    hbm_apb_pslverr_0 = 'b0;
  end

//=============================================================================
//
//=============================================================================

  always_comb begin
    PCIE_EP_TXP    = 'b0;
    PCIE_EP_TXN    = 'b0;

    PCIE_RP_PERSTN = 'b0;
    PCIE_RP_TXP    = 'b0;
    PCIE_RP_TXN    = 'b0;
  end

endmodule // cl_bittf
