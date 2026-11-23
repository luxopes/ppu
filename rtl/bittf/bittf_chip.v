// BitTF mini cip: jadro + radic QSPI PSRAM + ridici SPI (slave) pro hostitele (mikrokontroler).
//
// Vyvody (bez napajeni): clk, rst_n, PSRAM (sck, cs_n, io[3:0]), host_own, SPI (sck, cs_n, mosi, miso),
// done, busy = 15 signalu.
//
// Pamet: hostitel (napr. RP2040) ma sdilenou sbernici PSRAM. Pri host_own = 1 cip drzi vyvody PSRAM ve vysoke
// impedanci a hostitel primo nahraje program, vahy a korpus a cte vysledky; pak host_own = 0 a start pres SPI.
//
// SPI (rezim 0, MSB prvni, sck hostitele <= clk/8): ramec = cs_n v 0, bajt prikazu, 4 bajty dat.
//   prikaz bit7 = zapis, bity 2..0 = registr; pri cteni cip posila hodnotu registru v datovych bajtech.
//   0 seed, 1 ctr, 2 err, 3 cyc (CSR jadra; zapis jen v klidu), 4 prog_base,
//   5 rizeni: zapis bit0 = start; cteni {busy, done, 6'd0, err},
//   7 identifikace (cteni) = "BTF1".
module bittf_chip #(
    parameter AW = 24
) (
    input  wire       clk,
    input  wire       rst_n,
    // PSRAM
    output wire       psram_sck,
    output wire       psram_cs_n,
    output wire [3:0] psram_io_o,
    output wire [3:0] psram_io_oe,
    output wire       psram_ctl_oe,   // sck a cs_n (0 = vysoka impedance, sbernici ma hostitel)
    input  wire [3:0] psram_io_i,
    input  wire       host_own,
    // ridici SPI
    input  wire       spi_sck,
    input  wire       spi_cs_n,
    input  wire       spi_mosi,
    output reg        spi_miso,
    // stav
    output wire       done_o,
    output wire       busy_o
`ifdef BITTF_SIM
    ,
    output wire        dbg_instr_done,
    output wire [31:0] dbg_pc,
    input  wire [5:0]  dbg_ridx,
    output wire [63:0] dbg_rdata,
    output wire [31:0] dbg_ctr
`endif
);
    // ------------------------------------------------ reset a synchronizace vstupu
    reg [1:0] rst_s;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rst_s <= 2'b11;
        else rst_s <= {rst_s[0], 1'b0};
    wire rst = rst_s[1];

    reg [2:0] own_s, sck_s, cs_s, mosi_s;
    always @(posedge clk) begin
        own_s <= {own_s[1:0], host_own};
        sck_s <= {sck_s[1:0], spi_sck};
        cs_s <= {cs_s[1:0], spi_cs_n};
        mosi_s <= {mosi_s[1:0], spi_mosi};
    end
    wire own = own_s[1];
    wire sck_rise = sck_s[1] && !sck_s[2];
    wire sck_fall = !sck_s[1] && sck_s[2];
    wire cs_act = !cs_s[1];

    // ------------------------------------------------ jadro
    reg         start;
    reg  [31:0] prog_base;
    wire        busy, done;
    wire [7:0]  err;
    reg         csr_we;
    reg  [1:0]  csr_addr;
    reg  [31:0] csr_wdata;
    wire [31:0] csr_rdata;
    wire        m_req_valid, m_req_ready, m_req_we, m_rsp_valid;
    wire [31:0] m_req_addr;
    wire [1:0]  m_req_lw;
    wire [63:0] m_req_wdata, m_rsp_data;
    wire        c_dbg_instr_done;
    wire [31:0] c_dbg_pc;
    wire [63:0] c_dbg_rdata;
    wire [5:0]  c_dbg_ridx;
`ifdef BITTF_SIM
    assign dbg_instr_done = c_dbg_instr_done;
    assign dbg_pc = c_dbg_pc;
    assign dbg_rdata = c_dbg_rdata;
    assign c_dbg_ridx = dbg_ridx;
    assign dbg_ctr = core.csr_ctr;
`else
    assign c_dbg_ridx = 6'd0;
`endif

    bittf_mini #(.AW(AW), .CW(24)) core (
        .clk(clk), .rst(rst), .start(start), .prog_base(prog_base), .busy(busy), .done(done), .err(err),
        .csr_we(csr_we), .csr_addr(csr_addr), .csr_wdata(csr_wdata), .csr_rdata(csr_rdata),
        .m_req_valid(m_req_valid), .m_req_ready(m_req_ready), .m_req_we(m_req_we), .m_req_addr(m_req_addr),
        .m_req_lw(m_req_lw), .m_req_wdata(m_req_wdata), .m_rsp_valid(m_rsp_valid), .m_rsp_data(m_rsp_data),
        .dbg_instr_done(c_dbg_instr_done), .dbg_pc(c_dbg_pc), .dbg_ridx(c_dbg_ridx), .dbg_rdata(c_dbg_rdata)
    );

    wire q_oe;
    wire [3:0] q_io;
    bittf_qspi #(.AW(AW)) qspi (
        .clk(clk), .rst(rst), .host_own(own),
        .req_valid(m_req_valid), .req_ready(m_req_ready), .req_we(m_req_we), .req_addr(m_req_addr[AW-1:0]),
        .req_lw(m_req_lw), .req_wdata(m_req_wdata), .rsp_valid(m_rsp_valid), .rsp_data(m_rsp_data),
        .sck(psram_sck), .cs_n(psram_cs_n), .io_o(q_io), .io_oe(q_oe), .io_i(psram_io_i)
    );
    // ve vlastnictvi hostitele vse ve vysoke impedanci (hostitel si sbernici bere jen pri busy = 0)
    assign psram_io_o = q_io;
    assign psram_io_oe = {4{q_oe && !own}};
    assign psram_ctl_oe = !own;

    // ------------------------------------------------ ridici SPI
    reg        done_r;
    reg [5:0]  bitc;
    reg [39:0] sh;
    assign done_o = done_r;
    assign busy_o = busy;

    // hodnota registru pro cteni (cmd po prijmu prvniho bajtu)
    reg  [7:0]  cmd;
    reg  [31:0] rreg;
    always @(*) begin
        case (cmd[2:0])
            3'd0, 3'd1, 3'd2, 3'd3: rreg = csr_rdata;
            3'd4: rreg = prog_base;
            3'd5: rreg = {busy, done_r, 22'd0, err};
            3'd7: rreg = 32'h42544631;      // "BTF1"
            default: rreg = 32'd0;
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            bitc <= 6'd0;
            start <= 1'b0;
            prog_base <= 32'd0;
            csr_we <= 1'b0;
            done_r <= 1'b0;
            spi_miso <= 1'b0;
            cmd <= 8'd0;
        end else begin
            start <= 1'b0;
            csr_we <= 1'b0;
            if (done) done_r <= 1'b1;
            if (!cs_act) begin
                bitc <= 6'd0;
            end else begin
                if (sck_rise) begin
                    sh <= {sh[38:0], mosi_s[1]};
                    bitc <= bitc + 6'd1;
                    if (bitc == 6'd7)
                        cmd <= {sh[6:0], mosi_s[1]};
                    if (bitc == 6'd39 && cmd[7]) begin
                        // zapis registru
                        case (cmd[2:0])
                            3'd0, 3'd1, 3'd2, 3'd3: begin
                                csr_we <= 1'b1;
                                csr_addr <= cmd[1:0];
                                csr_wdata <= {sh[30:0], mosi_s[1]};
                            end
                            3'd4: prog_base <= {sh[30:0], mosi_s[1]};
                            3'd5: if (mosi_s[1] && !busy) begin start <= 1'b1; done_r <= 1'b0; end
                            default: ;
                        endcase
                    end
                end
                if (sck_fall) begin
                    // cteni: po prikazu (bitc 8..39) MSB prvni
                    if (bitc >= 6'd8 && bitc < 6'd40 && !cmd[7])
                        spi_miso <= rreg[6'd39 - bitc];
                    else
                        spi_miso <= 1'b0;
                end
                if (bitc == 6'd8 && !cmd[7])
                    csr_addr <= cmd[1:0];
            end
        end
    end
endmodule
