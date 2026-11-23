// Radic externi QSPI PSRAM (napr. APS6404L, 8 MiB) pro pametovy port jadra BitTF mini.
// PSRAM musi byt v rezimu QPI (prikaz 0x35 posle hostitel pri inicializaci, nez preda sbernici cipu).
//   cteni:  0xEB, 24bit adresa, WAIT cekacich taktu, 2 nibly na bajt (bajt na nizsi adrese prvni)
//   zapis:  0x38, 24bit adresa, 2 nibly na bajt
// SCK = clk/2, rezim 0 (klid v 0): vystupy se meni v nizke pulperiode, cteni se vzorkuje pri nabezne hrane SCK.
// host_own = 1: sbernici ma hostitel (vsechny vystupy ve vysoke impedanci), novy pozadavek se nepřijme.
// Jeden pozadavek naraz; odpoved na cteni vraci rsp_valid (data v dolnich bitech).
module bittf_qspi #(
    parameter AW = 24,
    parameter WAIT = 6
) (
    input  wire          clk,
    input  wire          rst,
    input  wire          host_own,
    // port jadra
    input  wire          req_valid,
    output wire          req_ready,
    input  wire          req_we,
    input  wire [AW-1:0] req_addr,
    input  wire [1:0]    req_lw,
    input  wire [63:0]   req_wdata,
    output reg           rsp_valid,
    output reg  [63:0]   rsp_data,
    // PSRAM
    output reg           sck,
    output reg           cs_n,
    output reg  [3:0]    io_o,
    output reg           io_oe,
    input  wire [3:0]    io_i
);
    localparam S_IDLE = 3'd0, S_TX = 3'd1, S_WAIT = 3'd2, S_RX = 3'd3, S_END = 3'd4, S_GAP = 3'd5;
    reg [2:0]  st;
    reg        ph;               // 0 = nizka pulperioda SCK, 1 = vysoka
    reg [95:0] tx;               // prikaz, adresa, data (MSB prvni)
    reg [4:0]  nc;               // pocitadlo niblu / cekacich taktu
    reg [4:0]  ntx, nrx;         // pocet niblu k vyslani / prijmu
    reg        we;
    reg [3:0]  nb;               // pocet bajtu (1, 2, 4, 8)
    reg [63:0] rd;               // prijata data, bajty se nasouvaji shora
    reg [3:0]  nib_hi;
    reg [1:0]  gap;

    assign req_ready = st == S_IDLE && !host_own;

    // data k zapisu: bajt 0 prvni
    function [63:0] bswap(input [63:0] v);
        bswap = {v[7:0], v[15:8], v[23:16], v[31:24], v[39:32], v[47:40], v[55:48], v[63:56]};
    endfunction

    wire [3:0] req_nb = 4'd1 << req_lw;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE;
            sck <= 1'b0;
            cs_n <= 1'b1;
            io_oe <= 1'b0;
            io_o <= 4'd0;
            rsp_valid <= 1'b0;
        end else begin
            rsp_valid <= 1'b0;
            case (st)
                S_IDLE: begin
                    sck <= 1'b0;
                    cs_n <= 1'b1;
                    io_oe <= 1'b0;
                    if (req_valid && req_ready) begin
                        we <= req_we;
                        nb <= req_nb;
                        tx <= {req_we ? 8'h38 : 8'hEB, {{(24 - AW){1'b0}}, req_addr}, bswap(req_wdata)};
                        ntx <= req_we ? 5'd8 + {req_nb, 1'b0} : 5'd8;
                        nrx <= {req_nb, 1'b0};
                        nc <= 5'd0;
                        ph <= 1'b0;
                        cs_n <= 1'b0;
                        st <= S_TX;
                    end
                end
                S_TX:
                    if (!ph) begin
                        sck <= 1'b0;
                        io_o <= tx[95:92];
                        io_oe <= 1'b1;
                        ph <= 1'b1;
                    end else begin
                        sck <= 1'b1;
                        tx <= tx << 4;
                        ph <= 1'b0;
                        if (nc == ntx - 5'd1) begin
                            nc <= 5'd0;
                            st <= we ? S_END : S_WAIT;
                        end else
                            nc <= nc + 5'd1;
                    end
                S_WAIT:
                    if (!ph) begin
                        sck <= 1'b0;
                        io_oe <= 1'b0;
                        ph <= 1'b1;
                    end else begin
                        sck <= 1'b1;
                        ph <= 1'b0;
                        if (nc == WAIT - 1) begin
                            nc <= 5'd0;
                            st <= S_RX;
                        end else
                            nc <= nc + 5'd1;
                    end
                S_RX:
                    if (!ph) begin
                        sck <= 1'b0;            // PSRAM vystavi nibl po sestupne hrane
                        ph <= 1'b1;
                    end else begin
                        sck <= 1'b1;            // vzorek pri nabezne hrane
                        ph <= 1'b0;
                        if (!nc[0])
                            nib_hi <= io_i;
                        else
                            rd <= {nib_hi, io_i, rd[63:8]};
                        if (nc == nrx - 5'd1) begin
                            nc <= 5'd0;
                            st <= S_END;
                        end else
                            nc <= nc + 5'd1;
                    end
                S_END: begin
                    sck <= 1'b0;
                    cs_n <= 1'b1;
                    io_oe <= 1'b0;
                    if (!we) begin
                        rsp_valid <= 1'b1;
                        case (nb)
                            4'd1: rsp_data <= {56'd0, rd[63:56]};
                            4'd2: rsp_data <= {48'd0, rd[63:48]};
                            4'd4: rsp_data <= {32'd0, rd[63:32]};
                            default: rsp_data <= rd;
                        endcase
                    end
                    gap <= 2'd0;
                    st <= S_GAP;
                end
                S_GAP: begin                    // CS# drzet v 1 alespon par taktu
                    gap <= gap + 2'd1;
                    if (gap == 2'd2) st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
