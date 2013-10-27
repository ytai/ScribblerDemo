import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.UIManager;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Main {
	static Random random_ = new Random();

	/**
	 * @param args
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws IOException, InterruptedException {
		if (args.length != 1) {
			System.exit(1);
		}

		Mat original = Highgui.imread(args[0]);
		Imgproc.cvtColor(original, original, Imgproc.COLOR_BGR2GRAY);
		Imgproc.resize(original, original, new Size(0, 0), 0.2, 0.2, Imgproc.INTER_LINEAR);
		
		Imgproc.equalizeHist(original, original);
		
		Mat in = new Mat(original.size(), CvType.CV_32FC1);
		original.convertTo(in, CvType.CV_32FC1);
		Core.multiply(in, new Scalar(1.0 / 255), in);
		Core.subtract(Mat.ones(in.size(), CvType.CV_32FC1), in, in);

		Mat out = randomLines(in, 0.2, 1000, 1000, true);

		out.convertTo(out, CvType.CV_8UC1);

		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
		}
		final JFrame frame = new JFrame("Scribble");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		Container contentPane = frame.getContentPane();
		contentPane.setLayout(new BoxLayout(contentPane, BoxLayout.X_AXIS));

		contentPane.add(new JLabel(new ImageIcon(matToBufferedImage(original))));
		ImageComponent component = new ImageComponent(matToBufferedImage(out));
		contentPane.add(component);

		frame.pack();
		frame.setVisible(true);
	}

	private static Mat randomLines(Mat in, double scale, int numLines, int numAttemptsPerLine, boolean continuous) {
		// Core.subtract(in, new Scalar(0.5), in);
		Core.multiply(in, new Scalar(1 / scale), in);

		Mat out = Mat.zeros(in.size(), CvType.CV_8U);
		Imgproc.resize(in, in, new Size(0, 0), scale, scale, Imgproc.INTER_CUBIC);
		Mat mask = Mat.zeros(in.size(), CvType.CV_8U);
		Mat outMask = Mat.zeros(out.size(), CvType.CV_8U);
		Mat bestMask = Mat.zeros(in.size(), CvType.CV_8U);
		Point lastP = null;
		int lines = 0;
		Point[] line = new Point[2];
		Point[] scaledLine = new Point[2];
		Point[] bestLine = null;
		while (lines < numLines) {
			double bestScore = Float.NEGATIVE_INFINITY;
			for (int i = 0; i < numAttemptsPerLine; ++i) {
				generateRandomLine(out.size(), continuous ? lastP : null, line);

				mask.setTo(new Scalar(0));
				scaleLine(line, scaledLine, scale);
				Core.line(mask, scaledLine[0], scaledLine[1], new Scalar(1));

				double score = Core.mean(in, mask).val[0];
				// double score = Core.mean(in, mask).val[0] *
				// Core.sumElems(mask).val[0];
				if (score > bestScore) {
					bestScore = score;
					Mat t = mask;
					mask = bestMask;
					bestMask = t;
					bestLine = line.clone();
				}
			}
			lastP = bestLine[1];

			outMask.setTo(new Scalar(0));
			Core.line(outMask, bestLine[0], bestLine[1], new Scalar(1));

			out.setTo(new Scalar(1), outMask);

			Core.subtract(in, bestMask, in, bestMask, in.type());
			++lines;
			System.out.format("%d -- score: %f, remaining: %f\n", lines, bestScore, Core.sumElems(in).val[0]);
		}

		Imgproc.threshold(out, out, 0.5, 255, Imgproc.THRESH_BINARY_INV);

		Core.normalize(in, in, 0, 255, Core.NORM_MINMAX);
		return out;
	}

	private static void scaleLine(Point[] line, Point[] scaledLine, double scale) {
		scaledLine[0] = new Point(line[0].x * scale, line[0].y * scale);
		scaledLine[1] = new Point(line[1].x * scale, line[1].y * scale);
	}

	private static void generateRandomLine(Size s, Point pStart, Point[] result) {
		if (pStart == null) {
			result[0] = new Point(random_.nextInt((int) s.width), random_.nextInt((int) s.height));
		} else {
			result[0] = pStart;
		}
		do {
			result[1] = new Point(random_.nextInt((int) s.width), random_.nextInt((int) s.height));
		} while (result[0].equals(result[1]));
	}

	static {
		System.loadLibrary("opencv_java246");
	}

	private static class ImageComponent extends Component {
		private static final long serialVersionUID = -8921722655371221897L;

		private final BufferedImage image_;
		private final List<List<int[]>> lines_ = new LinkedList<List<int[]>>();
		private boolean showImage_ = true;

		public ImageComponent(BufferedImage image) {
			image_ = image;
		}

		public synchronized void hideImage() {
			showImage_ = false;
			repaint();
		}

		public synchronized void addLine(List<int[]> line) {
			lines_.add(line);
			repaint();
		}

		@Override
		public Dimension getPreferredSize() {
			return new Dimension(image_.getWidth(), image_.getHeight());
		}

		@Override
		public synchronized void paint(Graphics g) {
			if (showImage_) {
				g.drawImage(image_, 0, 0, image_.getWidth(), image_.getHeight(), 0, 0, image_.getWidth(),
						image_.getHeight(), null);
			}
			g.setColor(Color.RED);
			for (List<int[]> line : lines_) {
				final Iterator<int[]> iter = line.iterator();
				int[] prev = iter.next();
				while (iter.hasNext()) {
					int next[] = iter.next();
					g.drawLine(prev[0], prev[1], next[0], next[1]);
					prev = next;
				}
			}
		}
	}

	public static BufferedImage matToBufferedImage(Mat matrix) {
		int cols = matrix.cols();
		int rows = matrix.rows();
		int elemSize = (int) matrix.elemSize();
		byte[] data = new byte[cols * rows * elemSize];
		int type;

		matrix.get(0, 0, data);

		switch (matrix.channels()) {
		case 1:
			type = BufferedImage.TYPE_BYTE_GRAY;
			break;

		case 3:
			type = BufferedImage.TYPE_3BYTE_BGR;

			// bgr to rgb
			byte b;
			for (int i = 0; i < data.length; i = i + 3) {
				b = data[i];
				data[i] = data[i + 2];
				data[i + 2] = b;
			}
			break;

		default:
			return null;
		}

		BufferedImage image = new BufferedImage(cols, rows, type);
		image.getRaster().setDataElements(0, 0, cols, rows, data);

		return image;
	}

}
