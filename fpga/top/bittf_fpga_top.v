`timescale 1ns/1ps
// BitTF na FPGA: jadro v2 + most na AXI4 (DDR/HBM) + registry pro hosta pres AXI4-Lite (OCL na AWS F2).
//
// Registry (offset bajtu):
//   0x00 CTRL       zapis bit0 = start programu
//   0x04 STATUS     bit0 busy, bit1 done (drzi do dalsiho startu), bity 15:8 kod chyby
//   0x08 PROG_BASE  adresa programu (v pameti jadra)
//   0x10..0x1C      CSR jadra: SEED, CTR, ERR, CYCLES (zapis jen kdyz jadro stoji)
//   0x20/0x24       MEM_BASE lo/hi: adresa AXI, kde zacina pamet jadra
//   0x28 ID         0x42544632 ("BTF2")
module bittf_fpga_top #(
    parameter LB = 64,
    parameter TM = 32,
    parameter TN = 32
) (
    input  wire          clk,
    input  wire          rst_n,
    // AXI4-Lite (OCL)
    input  wire [31:0]   s_axi_awaddr,
    input  wire          s_axi_awvalid,
    output wire          s_axi_awready,
    input  wire [31:0]   s_axi_wdata,
    input  wire [3:0]    s_axi_wstrb,
    input  wire          s_axi_wvalid,
    output wire          s_axi_wready,
    output wire [1:0]    s_axi_bresp,
    output wire          s_axi_bvalid,
    input  wire          s_axi_bready,
    input  wire [31:0]   s_axi_araddr,
    input  wire          s_axi_arvalid,
    output wire          s_axi_arready,
    output wire [31:0]   s_axi_rdata,
    output wire [1:0]    s_axi_rresp,
    output wire          s_axi_rvalid,
    input  wire          s_axi_rready,
    // AXI4 master do pameti
    output wire [63:0]   m_axi_araddr,
    output wire [7:0]    m_axi_arlen,
    output wire [2:0]    m_axi_arsize,
    output wire [1:0]    m_axi_arburst,
    output wire          m_axi_arvalid,
    input  wire          m_axi_arready,
    input  wire [LB*8-1:0] m_axi_rdata,
    input  wire          m_axi_rvalid,
    output wire          m_axi_rready,
    output wire [63:0]   m_axi_awaddr,
    output wire [7:0]    m_axi_awlen,
    output wire [2:0]    m_axi_awsize,
    output wire [1:0]    m_axi_awburst,
    output wire          m_axi_awvalid,
    input  wire          m_axi_awready,
    output wire [LB*8-1:0] m_axi_wdata,
    output wire [LB-1:0] m_axi_wstrb,
    output wire          m_axi_wlast,
    output wire          m_axi_wvalid,
    input  wire          m_axi_wready,
    input  wire          m_axi_bvalid,
    output wire          m_axi_bready,
    output wire          busy
);
    reg rst;
    always @(posedge clk) rst <= !rst_n;

    // --- MMIO
    wire        mmio_we, mmio_re;
    wire [7:0]  mmio_addr;
    wire [31:0] mmio_wdata;
    reg  [31:0] mmio_rdata;
    reg         mmio_rvalid;
    axi_lite_mmio #(.AW(8)) mmio (
        .aclk(clk), .aresetn(rst_n),
        .s_awaddr(s_axi_awaddr), .s_awvalid(s_axi_awvalid), .s_awready(s_axi_awready),
        .s_wdata(s_axi_wdata), .s_wstrb(s_axi_wstrb), .s_wvalid(s_axi_wvalid), .s_wready(s_axi_wready),
        .s_bresp(s_axi_bresp), .s_bvalid(s_axi_bvalid), .s_bready(s_axi_bready),
        .s_araddr(s_axi_araddr), .s_arvalid(s_axi_arvalid), .s_arready(s_axi_arready),
        .s_rdata(s_axi_rdata), .s_rresp(s_axi_rresp), .s_rvalid(s_axi_rvalid), .s_rready(s_axi_rready),
        .mmio_we(mmio_we), .mmio_re(mmio_re), .mmio_addr(mmio_addr), .mmio_wdata(mmio_wdata),
        .mmio_rdata(mmio_rdata), .mmio_rvalid(mmio_rvalid));

    reg         c_start, done_sticky;
    reg  [31:0] prog_base;
    reg  [63:0] mem_base;
    wire        c_busy, c_done;
    wire [7:0]  c_err;
    wire        csr_we = mmio_we && mmio_addr[7:4] == 4'h1;
    wire [31:0] csr_rdata;

    always @(posedge clk) begin
        if (rst) begin
            c_start <= 1'b0;
            done_sticky <= 1'b0;
            prog_base <= 0;
            mem_base <= 0;
            mmio_rvalid <= 1'b0;
        end else begin
            c_start <= 1'b0;
            mmio_rvalid <= mmio_re;
            if (c_done) done_sticky <= 1'b1;
            if (mmio_we) begin
                case (mmio_addr)
                    8'h00: if (mmio_wdata[0] && !c_busy) begin c_start <= 1'b1; done_sticky <= 1'b0; end
                    8'h08: prog_base <= mmio_wdata;
                    8'h20: mem_base[31:0] <= mmio_wdata;
                    8'h24: mem_base[63:32] <= mmio_wdata;
                    default: ;
                endcase
            end
            if (mmio_re) begin
                case (mmio_addr)
                    8'h04: mmio_rdata <= {16'd0, c_err, 6'd0, done_sticky, c_busy};
                    8'h08: mmio_rdata <= prog_base;
                    8'h10, 8'h14, 8'h18, 8'h1C: mmio_rdata <= csr_rdata;
                    8'h20: mmio_rdata <= mem_base[31:0];
                    8'h24: mmio_rdata <= mem_base[63:32];
                    8'h28: mmio_rdata <= 32'h42544632;
                    default: mmio_rdata <= 32'd0;
                endcase
            end
        end
    end

    // --- jadro + most
    wire [4:0] p_rq_valid, p_rq_ready, p_rq_we, p_rs_valid, p_wack;
    wire [5*26-1:0] p_rq_line;
    wire [5*LB*8-1:0] p_rq_wdata, p_rs_data;
    wire [5*LB-1:0] p_rq_wstrb;
    bittf_core2 #(.LB(LB), .TM(TM), .TN(TN)) core (
        .clk(clk), .rst(rst), .start(c_start), .prog_base(prog_base), .busy(c_busy), .done(c_done), .err(c_err),
        .csr_we(csr_we), .csr_addr(mmio_addr[3:2]), .csr_wdata(mmio_wdata), .csr_rdata(csr_rdata),
        .p_rq_valid(p_rq_valid), .p_rq_ready(p_rq_ready), .p_rq_we(p_rq_we), .p_rq_line(p_rq_line),
        .p_rq_wdata(p_rq_wdata), .p_rq_wstrb(p_rq_wstrb), .p_rs_valid(p_rs_valid), .p_rs_data(p_rs_data),
        .p_wack(p_wack), .dbg_instr_done(), .dbg_pc(), .dbg_ridx(6'd0), .dbg_rdata());
    bittf_axi #(.NP(5), .LB(LB)) axi (
        .clk(clk), .rst(rst), .mem_base(mem_base),
        .p_rq_valid(p_rq_valid), .p_rq_ready(p_rq_ready), .p_rq_we(p_rq_we), .p_rq_line(p_rq_line),
        .p_rq_wdata(p_rq_wdata), .p_rq_wstrb(p_rq_wstrb), .p_rs_valid(p_rs_valid), .p_rs_data(p_rs_data),
        .p_wack(p_wack),
        .m_araddr(m_axi_araddr), .m_arlen(m_axi_arlen), .m_arsize(m_axi_arsize), .m_arburst(m_axi_arburst),
        .m_arvalid(m_axi_arvalid), .m_arready(m_axi_arready), .m_rdata(m_axi_rdata), .m_rvalid(m_axi_rvalid),
        .m_rready(m_axi_rready), .m_awaddr(m_axi_awaddr), .m_awlen(m_axi_awlen), .m_awsize(m_axi_awsize),
        .m_awburst(m_axi_awburst), .m_awvalid(m_axi_awvalid), .m_awready(m_axi_awready), .m_wdata(m_axi_wdata),
        .m_wstrb(m_axi_wstrb), .m_wlast(m_axi_wlast), .m_wvalid(m_axi_wvalid), .m_wready(m_axi_wready),
        .m_bvalid(m_axi_bvalid), .m_bready(m_axi_bready));
    assign busy = c_busy;
endmodule
