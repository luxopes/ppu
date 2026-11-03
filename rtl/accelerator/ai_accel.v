`timescale 1ns/1ps
// Simple AI accelerator: sequential dot-product engine
// MMIO registers (offsets): CMD=0x00, STATUS=0x04, BASE1=0x08, BASE2=0x0C, LEN=0x10, RESULT=0x14
//
// Opravy 2026-11-03:
//  - po dokonceni se automat vraci do S_IDLE (drive zustal v S_DONE navzdy
//    a dalsi prikaz se uz nikdy neprovedl),
//  - vysledek obsahuje i posledni soucin (drive se ukladal acc pred prictenim),
//  - cteni pameti ceka na registrovana data (latence 2 takty; drive se bral
//    o takt starsi obsah sbernice),
//  - start z jadra je mozny i ve stavu "done" (STATUS=2), ne jen po resetu,
//  - pridan kombinacni port cpu_rdata pro cteni registru jednocyklovym jadrem.

module ai_accel(
    input  wire        clk,
    input  wire        rst,

    // optional MMIO interface
    input  wire        mmio_we,
    input  wire [7:0]  mmio_addr,
    input  wire [31:0] mmio_wdata,
    output reg  [31:0] mmio_rdata,
    // kombinacni cteni pro jadro (adresa z CPU)
    input  wire [7:0]  cpu_raddr,
    output reg  [31:0] cpu_rdata,
    // expose result directly
    output reg  [31:0] result_out,

    // memory read interface (data available 2 cycles after mem_rd_addr is set)
    output reg         mem_rd_en,
    output reg  [31:0] mem_rd_addr,
    input  wire [31:0] mem_rd_data,

    // direct control from core
    input  wire        start_in,
    input  wire [31:0] base1_in,
    input  wire [31:0] base2_in,
    input  wire [31:0] len_in
);

    // registers
    reg [31:0] reg_cmd;
    reg [31:0] reg_status; // 0 idle, 1 busy, 2 done
    reg [31:0] reg_base1;
    reg [31:0] reg_base2;
    reg [31:0] reg_len;
    reg signed [63:0] acc;

    reg [31:0] idx;
    reg [2:0] state;
    reg [31:0] tmp_v1;

    localparam S_IDLE    = 3'd0;
    localparam S_RD1     = 3'd1;
    localparam S_WAIT1   = 3'd2;
    localparam S_RD2     = 3'd3;
    localparam S_WAIT2   = 3'd4;
    localparam S_CAPTURE = 3'd5;

    wire signed [63:0] prod = $signed(tmp_v1) * $signed(mem_rd_data);

    // sequential FSM
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            reg_cmd <= 32'd0;
            reg_status <= 32'd0;
            reg_base1 <= 32'd0;
            reg_base2 <= 32'd0;
            reg_len <= 32'd0;
            acc <= 64'sd0;
            idx <= 32'd0;
            mem_rd_en <= 1'b0;
            mem_rd_addr <= 32'd0;
            result_out <= 32'd0;
            state <= S_IDLE;
            tmp_v1 <= 32'd0;
        end else begin
            // MMIO writes
            if (mmio_we) begin
                case (mmio_addr)
                    8'h00: reg_cmd <= mmio_wdata;
                    8'h04: reg_status <= mmio_wdata;
                    8'h08: reg_base1 <= mmio_wdata;
                    8'h0C: reg_base2 <= mmio_wdata;
                    8'h10: reg_len <= mmio_wdata;
                    default: ;
                endcase
            end

            // start via direct interface (pokud prave nebezi)
            if (start_in && reg_status != 32'd1) begin
                reg_base1 <= base1_in;
                reg_base2 <= base2_in;
                reg_len   <= len_in;
                reg_cmd   <= 32'd1;
                reg_status <= 32'd0;
            end

            // FSM
            case (state)
                S_IDLE: begin
                    mem_rd_en <= 1'b0;
                    if (reg_cmd == 32'd1 && reg_status != 32'd1 && !start_in) begin
                        reg_status <= 32'd1;
                        acc <= 64'sd0;
                        idx <= 32'd0;
                        if (reg_len == 32'd0) begin
                            // prazdny vektor: vysledek 0
                            reg_status <= 32'd2;
                            result_out <= 32'd0;
                            reg_cmd <= 32'd0;
                        end else begin
                            state <= S_RD1;
                        end
                    end
                end
                S_RD1: begin
                    mem_rd_en <= 1'b1;
                    mem_rd_addr <= reg_base1 + (idx << 2);
                    state <= S_WAIT1;
                end
                S_WAIT1: begin
                    mem_rd_en <= 1'b0;
                    state <= S_RD2;
                end
                S_RD2: begin
                    tmp_v1 <= mem_rd_data;           // prvni operand
                    mem_rd_en <= 1'b1;
                    mem_rd_addr <= reg_base2 + (idx << 2);
                    state <= S_WAIT2;
                end
                S_WAIT2: begin
                    mem_rd_en <= 1'b0;
                    state <= S_CAPTURE;
                end
                S_CAPTURE: begin
                    acc <= acc + prod;
                    idx <= idx + 1;
                    if (idx + 1 >= reg_len) begin
                        reg_status <= 32'd2;
                        result_out <= acc[31:0] + prod[31:0];
                        reg_cmd <= 32'd0;
                        state <= S_IDLE;
                    end else begin
                        state <= S_RD1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    function [31:0] reg_read;
        input [7:0] a;
        case (a)
            8'h00: reg_read = reg_cmd;
            8'h04: reg_read = reg_status;
            8'h08: reg_read = reg_base1;
            8'h0C: reg_read = reg_base2;
            8'h10: reg_read = reg_len;
            8'h14: reg_read = result_out;
            default: reg_read = 32'd0;
        endcase
    endfunction

    // MMIO read: registrovane (1 takt)
    always @(posedge clk) mmio_rdata <= reg_read(mmio_addr);
    always @* cpu_rdata = reg_read(cpu_raddr);

endmodule
